#pragma OPENCL EXTENSION cl_khr_fp64 : enable

constant uint NUM_FACES = 320;
constant uint NUM_VERTICES = 162;

// ----------------------------- Utility functions -----------------------------

float3 GetCOM(__global float3 *Verts, int ci) {
  float3 COM = (float3)(0.0f);
  for (int i = 0; i < NUM_VERTICES; i++) {
    COM += Verts[ci * NUM_VERTICES + i];
  }
  return COM / NUM_VERTICES;
}

float getVolume(const __global uint3 *VertIdxMat, __global float3 *Verts,
                int ci) {
  float volume = 0.0f;
  for (int i = 0; i < NUM_FACES; i++) {
    uint3 face = VertIdxMat[i];
    float3 vert0 = Verts[ci * NUM_VERTICES + face.x];
    float3 vert1 = Verts[ci * NUM_VERTICES + face.y];
    float3 vert2 = Verts[ci * NUM_VERTICES + face.z];
    float volumepart = dot(cross(vert1 - vert0, vert2 - vert0), vert0);
    volume += volumepart;
  }
  return fabs(volume) / 6.0f;
}

float3 applyPBC(float3 delta, float L, int PBC) {
  if (PBC)
    delta -= L * round(delta / L);
  return delta;
}

__kernel void VolumeForceUpdate(__global const uint3 *VertIdxMat,
                                __global float3 *Verts, __global float3 *Forces,
                                int NCELLS, __global float *Kv,
                                __global float *v0) {
  uint ci = get_global_id(1);
  uint fi = get_global_id(0);

  if (Kv[ci] == 0)
    return;

  uint3 vert_indices = VertIdxMat[fi];
  vert_indices += ci * NUM_VERTICES;

  float3 pos0 = Verts[vert_indices.x];
  float3 pos1 = Verts[vert_indices.y];
  float3 pos2 = Verts[vert_indices.z];

  float localVolume = getVolume(VertIdxMat, Verts, ci);
  float volumeStrain = (localVolume / v0[ci]) - 1.0f;

  float3 A = pos1 - pos0;
  float3 B = pos2 - pos0;
  float3 normal = normalize(cross(A, B));

  float third = 1.0f / 3.0f;
  Forces[vert_indices.x] -= Kv[ci] * third * volumeStrain * normal;
  Forces[vert_indices.y] -= Kv[ci] * third * volumeStrain * normal;
  Forces[vert_indices.z] -= Kv[ci] * third * volumeStrain * normal;
}

__kernel void SurfaceAreaForceUpdate(__global uint3 *VertIdxMat,
                                     __global float3 *Verts,
                                     __global float3 *Forces, uint NCELLS,
                                     __global float *Ka, __global float *l0) {
  uint ci = get_global_id(1);
  uint fi = get_global_id(0);
  if (Ka[ci] == 0)
    return;

  uint3 vert_indices = VertIdxMat[fi];
  vert_indices += ci * NUM_VERTICES;

  float3 pos0 = (Verts[vert_indices.x]);
  float3 pos1 = (Verts[vert_indices.y]);
  float3 pos2 = (Verts[vert_indices.z]);

  float3 lv0 = (pos1 - pos0);
  float3 lv1 = (pos2 - pos1);
  float3 lv2 = (pos0 - pos2);

  float l0v0 = length(lv0);
  float l0v1 = length(lv1);
  float l0v2 = length(lv2);

  float3 ulv0 = lv0 / l0v0;
  float3 ulv1 = lv1 / l0v1;
  float3 ulv2 = lv2 / l0v2;

  float3 dli = (float3)(l0v0 / l0[ci] - 1.0f, l0v1 / l0[ci] - 1.0f,
                        l0v2 / l0[ci] - 1.0f);

  float third = 1.0f / 3.0f;
  Forces[vert_indices.x] += third * Ka[ci] * (dli.x * ulv0 - dli.z * ulv2);
  Forces[vert_indices.y] += third * Ka[ci] * (dli.y * ulv1 - dli.x * ulv0);
  Forces[vert_indices.z] += third * Ka[ci] * (dli.z * ulv2 - dli.y * ulv1);
}

__kernel void StickToSurface(__global uint3 *VertIdxMat, __global float3 *Verts,
                             __global float3 *Forces, uint NCELLS,
                             __global float *Ks, __global float *l0) {
  uint ci = get_global_id(1);
  uint fi = get_global_id(0);
  if (Ks[ci] == 0)
    return;

  uint3 vert_indices = VertIdxMat[fi];
  vert_indices += ci * NUM_VERTICES;

  float3 pos0 = Verts[vert_indices.x];
  float3 pos1 = Verts[vert_indices.y];
  float3 pos2 = Verts[vert_indices.z];

  float3 COM = GetCOM(Verts, ci);

  if (pos0.z < l0[ci] && pos1.z < l0[ci] && pos2.z < l0[ci]) {
    float third = 1.0f / 3.0f;
    Forces[vert_indices.x] += third * Ks[ci] * normalize(pos0 - COM);
    Forces[vert_indices.y] += third * Ks[ci] * normalize(pos1 - COM);
    Forces[vert_indices.z] += third * Ks[ci] * normalize(pos2 - COM);
  }

  // Push vertices below plane
  if (pos0.z < 0.0f)
    Forces[vert_indices.x].z -= Ks[ci] * pos0.z / l0[ci] / 3.0f;
  if (pos1.z < 0.0f)
    Forces[vert_indices.y].z -= Ks[ci] * pos1.z / l0[ci] / 3.0f;
  if (pos2.z < 0.0f)
    Forces[vert_indices.z].z -= Ks[ci] * pos2.z / l0[ci] / 3.0f;
}

__kernel void RepellingForces(__global uint3 *VertIdxMat,
                              __global float3 *Verts, __global float3 *Forces,
                              uint NCELLS, __global float *l0, float Kc,
                              int PBC, float L) {
  uint ci = get_global_id(1); // current cell
  uint vi = get_global_id(0); // current vertex
  uint vert_index = ci * NUM_VERTICES + vi;

  float3 p = Verts[vert_index];
  float3 pf = (float3)(p.x, p.y, p.z);

  for (uint cj = 0; cj < NCELLS; cj++) {
    if (ci == cj)
      continue;

    // compute COM of other cell
    float3 COM = (float3)(0.0f);
    for (uint k = 0; k < NUM_VERTICES; k++) {
      float3 v = Verts[cj * NUM_VERTICES + k];
      if (PBC) {
        float3 delta = (float3)(v.x, v.y, v.z) - pf;
        delta -= L * round(delta / L);
        v.x = pf.x + delta.x;
        v.y = pf.y + delta.y;
        v.z = pf.z + delta.z;
      }
      COM += v;
    }
    COM /= NUM_VERTICES;
    float3 COMf = (float3)(COM.x, COM.y, COM.z);

    // Compute winding number (solid angle) for this vertex relative to cell cj
    float winding = 0.0f;
    for (uint fi = 0; fi < NUM_FACES; fi++) {
      uint3 face = VertIdxMat[fi];
      float3 v0 = Verts[cj * NUM_VERTICES + face.x];
      float3 v1 = Verts[cj * NUM_VERTICES + face.y];
      float3 v2 = Verts[cj * NUM_VERTICES + face.z];

      float3 r0 = (float3)(v0.x, v0.y, v0.z) - pf;
      float3 r1 = (float3)(v1.x, v1.y, v1.z) - pf;
      float3 r2 = (float3)(v2.x, v2.y, v2.z) - pf;

      if (PBC) {
        r0 -= L * round(r0 / L);
        r1 -= L * round(r1 / L);
        r2 -= L * round(r2 / L);
      }

      float r0_len = length(r0);
      float r1_len = length(r1);
      float r2_len = length(r2);

      float triple = dot(r0, cross(r1, r2));
      float denom = r0_len * r1_len * r2_len + dot(r0, r1) * r2_len +
                    dot(r1, r2) * r0_len + dot(r2, r0) * r1_len;
      winding += atan2(triple, denom);
    }
    winding /= (4.0f * M_PI_F); // normalize

    // if vertex is inside other cell, apply repelling force
    if (winding > 0.5f) {
      float3 dir = normalize(pf - COMf);
      float3 f = Kc * dir;
      Forces[vert_index] += (float3)(f.x, f.y, f.z);
    }
  }
}

__kernel void AllVertAttraction(__global float3 *Verts, __global float3 *Forces,
                                __global float *l0, float L, int NCELLS,
                                int PBC, float Kat) {
  if (Kat == 0)
    return;

  uint ci = get_global_id(1);
  uint vi = get_global_id(0);
  uint vert_index = ci * NUM_VERTICES + vi;
  float3 p1 = Verts[vert_index];
  float3 pf1 = (float3)(p1.x, p1.y, p1.z);

  for (int cj = 0; cj < NCELLS; cj++) {
    if (ci == cj)
      continue;
    for (int vj = 0; vj < NUM_VERTICES; vj++) {
      float3 p2 = Verts[cj * NUM_VERTICES + vj];
      float3 delta = (float3)(p2.x, p2.y, p2.z) - pf1;
      delta = applyPBC(delta, L, PBC);
      float dist = length(delta);
      if (dist < l0[ci]) {
        float3 f = Kat * 0.5f * (dist / l0[ci] - 1.0f) * normalize(delta);
        Forces[vert_index] -= (float3)(f.x, f.y, f.z);
        Forces[cj * NUM_VERTICES + vj] += (float3)(f.x, f.y, f.z);
      }
    }
  }
}

__kernel void EulerPosition(__global float3 *Verts, __global float3 *Forces,
                            float dt) {
  uint ci = get_global_id(1);
  uint vi = get_global_id(0);
  uint vert_index = ci * NUM_VERTICES + vi;
  Verts[vert_index] += Forces[vert_index] * dt;
  Forces[vert_index] = (float3)(0.0f);
}
