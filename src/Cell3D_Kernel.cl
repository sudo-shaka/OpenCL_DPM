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
__kernel void VolumeForceUpdate(__global const uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, int NCELLS, float v0, float Kv){

  /*data is structred like
      cell1, cell2, cell3, cell4, ...
    f1
    f2
    f3
    ...
  */
  int NUM_FACES = 320; //number of faces
  int NUM_VERTICES = 162; //number of vertices

  int ci  = get_global_id(0); //cell index
  int fi  = get_global_id(1); //face index at that cell

  int face_index = ci * NCELLS + fi;
  uint4 vert_indicies = VertIdxMat[face_index];

  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  float determinate =
  (pos0[0] * pos1[1] * pos2[2]) +
  (pos0[1] * pos1[2] * pos2[0]) +
  (pos0[2] * pos1[0] * pos2[1]) -
  (pos0[2] * pos1[1] * pos2[0]) -
  (pos0[1] * pos1[0] * pos2[2]) -
  (pos0[0] * pos1[2] * pos2[1]);
  if (determinate < 0.0){
    determinate = -determinate;
  }
  __local float local_volume_sum[320];
  local_volume_sum[fi] = determinate;

  barrier(CLK_LOCAL_MEM_FENCE);

  float volume = 0.0;
  for (int i = 0; i < NUM_FACES; i++){
    volume += local_volume_sum[i];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  volume /= 6;

  float volumeStrain = (volume/v0) - 1.0;
  float4 force0 = (float4)(0.0f);
  float4 force1 = (float4)(0.0f);
  float4 force2 = (float4)(0.0f);

  float4 A = pos1 - pos0;
  float4 B = pos2 - pos0;

  float4 normal = normalize(cross(A, B));
  Forces[vert_indicies[0]] -= Kv * 0.5f * volumeStrain * normal;
  Forces[vert_indicies[1]] -= Kv * 0.5f * volumeStrain * normal;
  Forces[vert_indicies[2]] -= Kv * 0.5f * volumeStrain * normal;
}

__kernel void SurfaceAreaForceUpdate(__global uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, int NCELLS, float sa0, float Ka){
  int NUM_FACES = 320; //number of faces
  int NUM_VERTICES = 162; //number of vertices

  int ci  = get_global_id(0); //cell index
  int fi  = get_global_id(1); //face index at that cell

  int face_index = ci * NCELLS + fi;
  uint4 vert_indicies = VertIdxMat[face_index];

  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  float4 lv0 = pos1 - pos0;
  float4 lv1 = pos2 - pos1;
  float4 lv2 = pos0 - pos2;

  float4 lenghs = (float4)(sqrt(dot(lv0,lv0)), sqrt(dot(lv1,lv1)), sqrt(dot(lv2,lv2)), 0.0f);
}

__kernel void UpdatePosition(__global float4* Verts, __global float4* Forces, int NCELLS ,float dt){
  int ci = get_global_id(0);
  int vi = get_global_id(1);

  int vert_index = ci * NCELLS + vi;
  Verts[vert_index] += Forces[vert_index] * dt;
}
