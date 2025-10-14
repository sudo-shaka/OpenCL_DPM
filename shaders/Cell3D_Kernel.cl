
// Fixed OpenCL kernels and helpers
// Assumes OpenCL 1.2-ish vector functions; adjust if your platform differs.

constant uint NUM_FACES = 320;
constant uint NUM_VERTICES = 162;
constant float PI = 3.14159265358979323846f;

// --- Atomic float add (compare-and-swap) ---
void atomic_add_f(__global float *addr, float val) {
  union {
    unsigned int i;
    float f;
  } oldv, newv;
  unsigned int expected;
  do {
    // read current value
    oldv.f = *addr;
    newv.f = oldv.f + val;
    expected = oldv.i;
    // try to swap
  } while (atomic_cmpxchg((volatile __global int *)addr, (int)expected,
                          (int)newv.i) != (int)expected);
}

// Atomic add for float3 (adds x,y,z,w components individually)
void atomic_add_float3(__global float3 *addr, float3 val) {
  __global float *f = (__global float *)addr;
  atomic_add_f(&f[0], val.x);
  atomic_add_f(&f[1], val.y);
  atomic_add_f(&f[2], val.z);
}

// Simple center-of-mass for a cell (average of its vertices)
float3 GetCOM(__global const float3 *Verts, int ci) {
  float3 sum = (float3)(0.0f, 0.0f, 0.0f);
  int base = ci * NUM_VERTICES;
  for (uint i = 0; i < NUM_VERTICES; ++i) {
    sum += Verts[base + i];
  }
  float inv = 1.0f / (float)NUM_VERTICES;
  sum *= inv;
  return sum;
}

float getVolume(const __global uint3 *VertIdxMat, __global float3 *Verts,
                uint ci) {
  float volume = 0.0f;

  for (uint fi = 0; fi < NUM_FACES; fi++) {
    uint3 face = VertIdxMat[fi];
    int i0 = ci * NUM_VERTICES + face.x;
    int i1 = ci * NUM_VERTICES + face.y;
    int i2 = ci * NUM_VERTICES + face.z;

    float3 P0 = (float3)(Verts[i0].x, Verts[i0].y, Verts[i0].z);
    float3 P1 = (float3)(Verts[i1].x, Verts[i1].y, Verts[i1].z);
    float3 P2 = (float3)(Verts[i2].x, Verts[i2].y, Verts[i2].z);

    volume += dot(cross(P0, P1), P2) / 6.0f;
  }

  return fabs(volume);
}

__kernel void VolumeForceUpdate(__global const uint3 *VertIdxMat,
                                __global float3 *Verts, __global float3 *Forces,
                                int NCELLS, __global float *Kv,
                                __global float *v0,
                                __global float *cellVolumes) {
  uint fi = get_global_id(0);
  uint ci = get_global_id(1);

  if (ci >= NCELLS || fi >= NUM_FACES || Kv[ci] == 0.0f)
    return;

  // Compute volume ONCE per cell (thread fi == 0)
  if (fi == 0) {
    cellVolumes[ci] = getVolume(VertIdxMat, Verts, ci);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  float volume = cellVolumes[ci];
  float volumeStrain = (volume / v0[ci]) - 1.0f;

  uint3 face = VertIdxMat[fi];
  int i0 = ci * NUM_VERTICES + face.x;
  int i1 = ci * NUM_VERTICES + face.y;
  int i2 = ci * NUM_VERTICES + face.z;

  float3 P0 = Verts[i0];
  float3 P1 = Verts[i1];
  float3 P2 = Verts[i2];

  float3 COM = GetCOM(Verts, ci);
  float3 A = P1 - COM;
  float3 B = P2 - COM;
  float3 C = P0 - COM;

  float3 grad0 = cross(A, B);
  float3 grad1 = cross(B, C);
  float3 grad2 = cross(C, A);

  float Kv_val = Kv[ci];
  float3 f0 = -Kv_val * volumeStrain * grad0 / 6.0f;
  float3 f1 = -Kv_val * volumeStrain * grad1 / 6.0f;
  float3 f2 = -Kv_val * volumeStrain * grad2 / 6.0f;

  atomic_add_float3(&Forces[i0], f0);
  atomic_add_float3(&Forces[i1], f1);
  atomic_add_float3(&Forces[i2], f2);
}

__kernel void
SurfaceAreaForceUpdate(__global const uint3 *VertIdxMat, __global float3 *Verts,
                       __global float3 *Forces, uint NCELLS,
                       __global float *Ka, // surface area stiffness
                       __global float *a0, // target surface area
                       __global float *l0  // target edge length per cell
) {
  uint fi = get_global_id(0); // face index
  uint ci = get_global_id(1); // cell index

  if (ci >= NCELLS || fi >= NUM_FACES)
    return;
  if (Ka[ci] < 1e-8f)
    return;

  uint3 face = VertIdxMat[fi];
  int i0 = ci * NUM_VERTICES + face.x;
  int i1 = ci * NUM_VERTICES + face.y;
  int i2 = ci * NUM_VERTICES + face.z;

  // Get vertex positions (pos0, pos1, pos2)
  float3 pos0 = Verts[i0];
  float3 pos1 = Verts[i1];
  float3 pos2 = Verts[i2];

  // Calculate edge vectors (lv0, lv1, lv2)
  float3 lv0 = pos1 - pos0; // lv0 := subVector(pos1, pos0)
  float3 lv1 = pos2 - pos1; // lv1 := subVector(pos2, pos1)
  float3 lv2 = pos0 - pos2; // lv2 := subVector(pos0, pos2)

  // Calculate edge lengths
  float lengths[3] = {
      length(lv0), // distance3D(pos1, pos0)
      length(lv1), // distance3D(pos2, pos1)
      length(lv2)  // distance3D(pos0, pos2)
  };

  // Avoid division by zero
  if (lengths[0] < 1e-12f || lengths[1] < 1e-12f || lengths[2] < 1e-12f)
    return;

  // Normalize edge vectors (lv0.divideBy(lengths[0]), etc.)
  float3 lv0_norm = lv0 / lengths[0];
  float3 lv1_norm = lv1 / lengths[1];
  float3 lv2_norm = lv2 / lengths[2];

  // Calculate strain for each edge (dli)
  float target_length = l0[ci];
  float3 dli = (float3)((lengths[0] / target_length) - 1.0f, // dli.X
                        (lengths[1] / target_length) - 1.0f, // dli.Y
                        (lengths[2] / target_length) - 1.0f  // dli.Z
  );

  float scale = Ka[ci] * sqrt(a0[ci]) / target_length * 0.3f;
  float3 f0 = (lv0_norm * dli.x) - (lv2_norm * dli.z);
  float3 f1 = (lv1_norm * dli.y) - (lv0_norm * dli.x);
  float3 f2 = (lv2_norm * dli.z) - (lv1_norm * dli.y);
  f0 *= scale;
  f1 *= scale;
  f2 *= scale;
  atomic_add_float3(&Forces[i0], f0);
  atomic_add_float3(&Forces[i1], f1);
  atomic_add_float3(&Forces[i2], f2);
}

// ---------------- StickToSurface ----------------
__kernel void StickToSurface(__global const uint3 *VertIdxMat,
                             __global const float3 *Verts,
                             __global float3 *Forces, uint NCELLS,
                             __global const float *Ks,
                             __global const float *l0) {

  uint ci = get_global_id(1);
  uint fi = get_global_id(0);

  if (ci >= NCELLS)
    return;
  if (fi >= NUM_FACES)
    return;
  if (Ks[ci] < 1e-12f)
    return;

  uint3 face = VertIdxMat[fi];
  int base = ci * NUM_VERTICES;
  int i0 = base + face.x;
  int i1 = base + face.y;
  int i2 = base + face.z;

  float3 P0 = Verts[i0];
  float3 P1 = Verts[i1];
  float3 P2 = Verts[i2];

  // Calculate face normal
  float3 A = P1 - P0;
  float3 B = P2 - P0;
  float3 normal = cross(A, B);
  float3 unit_normal = normalize(normal);

  // Check if face is pointing downward (toward z=0 surface)
  // Negative z-component means face normal points down toward surface
  bool facingDownward = unit_normal.z < -0.1f; // Threshold for "facing surface"

  if (!facingDownward)
    return; // Only apply forces to faces facing the surface

  float3 COM = GetCOM(Verts, ci);
  float ks = Ks[ci];
  float target_length = l0[ci];

  float3 positions[3] = {P0, P1, P2};
  int indices[3] = {i0, i1, i2};

  for (int i = 0; i < 3; i++) {
    float3 pos = positions[i];

    // Keep vertices above surface (z >= 0)
    if (pos.z < 0.0f) {
      float z_force = ks * fabs(pos.z);
      atomic_add_f(&((__global float *)&Forces[indices[i]])[2], z_force);
    }

    // Create flattened position on surface
    float3 surface_pos = (float3)(pos.x, pos.y, 0.0f);
    float height = fabs(pos.z);

    if (height < l0[ci] * 2.0f) { // Adjust threshold as needed

      float3 com_to_vertex = surface_pos - COM;
      float ftmp = Ks[ci] * (1.0 - distance(pos, surface_pos) / l0[ci]);
      float3 force = normalize(com_to_vertex) * ftmp;
      atomic_add_float3(&Forces[indices[i]], force);
    }
  }
}

// ---------------- Repelling Forces (winding number based) ----------------

__kernel void RepellingForces(__global const uint3 *VertIdxMat,
                              __global const float3 *Verts,
                              __global float3 *Forces, uint NCELLS,
                              __global const float *l0, float Kc, int PBC,
                              float L) {
  uint global_vi = get_global_id(0); // Global vertex index

  if (global_vi >= NCELLS * NUM_VERTICES)
    return;
  if (Kc == 0.0f)
    return;

  uint ci = global_vi / NUM_VERTICES;       // Which cell this vertex belongs to
  uint local_vi = global_vi % NUM_VERTICES; // Local vertex index within cell

  float3 point = Verts[global_vi];
  float3 COMi = GetCOM(Verts, ci);

  for (int cj = 0; cj < NCELLS; cj++) {
    if (cj == ci)
      continue;

    float totalOmega = 0.0f;
    float3 COMJ = GetCOM(Verts, cj);

    // PBC shift
    float3 shift = (float3)(0, 0, 0);
    if (PBC) {
      shift = L * round((COMi - COMJ) / L);
      COMJ += shift;
    }

    for (uint fj = 0; fj < NUM_FACES; fj++) {
      uint3 vj = VertIdxMat[fj];
      float3 a = Verts[cj * NUM_VERTICES + vj.x] + shift - point;
      float3 b = Verts[cj * NUM_VERTICES + vj.y] + shift - point;
      float3 c = Verts[cj * NUM_VERTICES + vj.z] + shift - point;

      float3 u = normalize(a);
      float3 v = normalize(b);
      float3 w = normalize(c);

      float denom = 1.0f + dot(u, v) + dot(v, w) + dot(w, u);
      if (denom < 1e-8f)
        continue;

      float num = dot(u, cross(v, w));
      float omega = 2.0f * atan2(num, denom);
      totalOmega += omega;
    }

    float windingNumber = totalOmega / (4.0f * M_PI_F);
    if (fabs(windingNumber) < 1e-6f)
      continue;

    float3 dir = normalize(COMi - point);
    float3 tmpForce = fabs(windingNumber) * 0.5f * Kc * dir;
    atomic_add_float3(&Forces[global_vi], tmpForce);
  }
}

// ---------------- All Vertex Attraction ----------------
__kernel void AllVertAttraction(__global const float3 *Verts,
                                __global float3 *Forces,
                                __global const float *l0, float L, int NCELLS,
                                int PBC, float Kat) {
  if (Kat == 0.0f)
    return;

  uint ci = get_global_id(1);
  uint vi = get_global_id(0);

  if (ci >= (uint)NCELLS)
    return;
  if (vi >= NUM_VERTICES)
    return;

  int base = ci * NUM_VERTICES;
  int vert_index = base + vi;
  float3 p1 = Verts[vert_index];

  for (int cj = 0; cj < NCELLS; ++cj) {
    if (cj == (int)ci)
      continue;
    int basej = cj * NUM_VERTICES;
    for (int vj = 0; vj < (int)NUM_VERTICES; ++vj) {
      float3 p2 = Verts[basej + vj];
      float3 delta4 = p2 - p1;
      float3 delta = (float3)(delta4.x, delta4.y, delta4.z);

      if (PBC) {
        float rx = round(delta.x / L);
        float ry = round(delta.y / L);
        float rz = round(delta.z / L);
        delta -= (float3)(L * rx, L * ry, L * rz);
      }

      float dist = length(delta);
      float L0 = l0[ci];
      if (L0 < 1e-12f)
        L0 = 1e-12f;

      if (dist < L0 * 2.0f && dist > 1e-12f) {
        float3 tmpForce =
            (float3)(Kat * 0.5f * (dist / L0 - 1.0f) * (delta.x / dist),
                     Kat * 0.5f * (dist / L0 - 1.0f) * (delta.y / dist),
                     Kat * 0.5f * (dist / L0 - 1.0f) * (delta.z / dist));
        atomic_add_float3(&Forces[vert_index],
                          (float3)(-tmpForce.x, -tmpForce.y, -tmpForce.z));
        atomic_add_float3(&Forces[basej + vj], tmpForce);
      }
    }
  }
}

__kernel void ClearForces(__global float3 *Forces) {
  uint idx = get_global_id(0);
  Forces[idx] = (float3)(0.0f, 0.0f, 0.0f);
}

__kernel void EulerPosition(__global float3 *Verts, __global float3 *Forces,
                            float dt, uint NCELLS) {
  uint ci = get_global_id(1);
  uint vi = get_global_id(0);

  if (ci >= NCELLS || vi >= NUM_VERTICES)
    return;

  uint vert_index = ci * NUM_VERTICES + vi;
  Verts[vert_index] += Forces[vert_index] * dt;
}
