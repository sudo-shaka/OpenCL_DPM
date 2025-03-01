/*
 * Updates the volume forces acting on the vertices of a 3D cell structure.
 *
 * @param VertIdxMat  Global memory pointer to the vertex index matrix.
 * @param Verts       Global memory pointer to the vertex positions.
 * @param Forces      Global memory pointer to the forces acting on the vertices.
 * @param NCELLS      Number of cells.
 * @param v0          Reference volume.
 * @param Kv          Volume stiffness coefficient.
  int NUM_FACES = 320; //number of faces
  int NUM_VERTICIES = 162; //number of vertices
*/

const int NUM_FACES = 320;
const int NUM_VERTICES = 162;
float volume_sum[NUM_FACES];
float4 position_sum[NUM_VERTICES];

float4 GetCOM(__global float4* Verts ,int ci){
  float4 COM = (float4)(0.0f,0.0f,0.0f,0.0f);
  for(int i = 0; i < NUM_VERTICES; i++){
    COM += Verts[ci * NUM_VERTICES + i];
  }
  return COM/NUM_VERTICES;
}

__kernel void VolumeForceUpdate(__global const uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, int NCELLS, float Kv, float v0){

  /*data is structred like
      cell1, cell2, cell3, cell4, ...
    f1
    f2
    f3
    ...
  */
  int ci  = get_global_id(0); //cell index
  int fi  = get_global_id(1); //face index at that cell

  int face_index = ci * NCELLS + fi;
  uint4 vert_indicies = VertIdxMat[face_index];
  // Get the positions of the vertices of the current face
  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  // Calculate the volume contribution of the current face using the scalar triple product
  volume_sum[fi] = (dot(cross(pos1, pos2), pos0));
  if (volume_sum[fi] < 0)
    volume_sum[fi] *= -1; // Ensure the volume contribution is positive

  barrier(CLK_LOCAL_MEM_FENCE);

  // Sum up the volume contributions from all faces
  float volume = 0.0;
  for (int i = 0; i < NUM_FACES; i++) {
    volume += volume_sum[i] / 6;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Calculate the volume strain
  float volumeStrain = (volume / v0) - 1.0;

  // Initialize force vectors for the vertices
  float4 force0 = (float4)(0.0f);
  float4 force1 = (float4)(0.0f);
  float4 force2 = (float4)(0.0f);

  // Calculate the normal vector of the face
  float4 A = pos1 - pos0;
  float4 B = pos2 - pos0;
  float4 normal = normalize(cross(A, B));

  // Update the forces acting on the vertices based on the volume strain
  Forces[vert_indicies[0]] -= Kv * 1.0f / 3.0f * volumeStrain * normal;
  Forces[vert_indicies[1]] -= Kv * 1.0f / 3.0f * volumeStrain * normal;
  Forces[vert_indicies[2]] -= Kv * 1.0f / 3.0f * volumeStrain * normal;
}

__kernel void SurfaceAreaForceUpdate(__global uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, int NCELLS, float Ka, float l0){
  int ci  = get_global_id(0); //cell index
  int fi  = get_global_id(1); //face index at that cell

  int face_index = ci * NCELLS + fi;
  uint4 vert_indicies = VertIdxMat[face_index];

  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  // Calculate the lengths of the edges of the triangle
  float4 lv0 = pos1 - pos0;
  float4 lv1 = pos2 - pos1;
  float4 lv2 = pos0 - pos2;
  float4 lengths = (float4)(sqrt(dot(lv0, lv0)), sqrt(dot(lv1, lv1)), sqrt(dot(lv2, lv2)), 0.0f);

  // Calculate the unit vectors for the edges
  float4 ulv0 = lv0 / lengths[0];
  float4 ulv1 = lv1 / lengths[1];
  float4 ulv2 = lv2 / lengths[2];

  // Calculate the length strain for each edge
  float4 dli = (float4)(lengths[0]/l0 - 1.0, lengths[1]/l0 - 1.0, lengths[2]/l0 - 1.0, 0.0);

  // Calculate the force contributions for each vertex
  float third = 1.0 / 3.0;
  Forces[vert_indicies[0]] += third * Ka * (dli[0] * ulv0 - dli[2] * ulv2);
  Forces[vert_indicies[1]] += third * Ka * (dli[1] * ulv1 - dli[0] * ulv0);
  Forces[vert_indicies[2]] += third * Ka * (dli[2] * ulv2 - dli[1] * ulv1);
}

__kernel void StickToSurface(__global uint4* VertIdxMat,__global float4* Verts, __global float4* Forces, int NCELLS, float Ks, float l0){
  int ci = get_global_id(0);
  int fi = get_global_id(1);
  int face_index = ci * NCELLS + fi;
  uint4 vert_indicies = VertIdxMat[face_index];

  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  float4 A = pos1 - pos0;
  float4 B = pos2 - pos0;
  bool isnormal = ((A[0]*B[1] - A[1]*B[0]) < 0.0f);

  float4 surfacePoint0 = (float4)(pos0[0], pos0[1], 0.0f, 0.0f);
  float4 surfacePoint1 = (float4)(pos1[0], pos1[1], 0.0f, 0.0f);
  float4 surfacePoint2 = (float4)(pos2[0], pos2[1], 0.0f, 0.0f);

  int4 isunder = (int4)(0,0,0,0);

  if(pos0[2] < 0.0f){
    Forces[vert_indicies[0]][2] -= Ks * (pos0[2]/l0) * (1.0f/3.0f);
    isunder[0] = 1;
}
  if(pos1[2] < 0.0f){
    Forces[vert_indicies[1]][2] -= Ks * (pos1[2]/l0) * (1.0f/3.0f);
    isunder[1] = 1;
  }
  if(pos2[2] < 0.0f){
    Forces[vert_indicies[2]][2] -= Ks * (pos2[2]/l0) * (1.0f/3.0f);
    isunder[2] = 1;
  }

  float4 COM = GetCOM(Verts, ci);
  barrier(CLK_LOCAL_MEM_FENCE);

  if(isnormal && !isunder[0] && pos0[2] < l0){
    Forces[vert_indicies[0]] += (1.0f/3.0f) * Ks * ((1.0f - pos0[2])/l0) * normalize(surfacePoint0 - COM);
  }
  if(isnormal && !isunder[1] && pos1[2] < l0){
    Forces[vert_indicies[1]] += (1.0f/3.0f) * Ks * ((1.0f - pos1[2])/l0) * normalize(surfacePoint1 - COM);
  }
  if(isnormal && !isunder[2] && pos2[2] < l0){
    Forces[vert_indicies[2]] += (1.0f/3.0f) * Ks * ((1.0f - pos2[2])/l0) * normalize(surfacePoint2 - COM);
  }

  
}

__kernel void EulerPosition(__global float4* Verts, __global float4* Forces, int NCELLS ,float dt){
  int ci = get_global_id(0);
  int vi = get_global_id(1);

  int vert_index = ci * NCELLS + vi;
  Verts[vert_index] += Forces[vert_index] * dt;
  Forces[vert_index] = (float4)(0.0f);
}
