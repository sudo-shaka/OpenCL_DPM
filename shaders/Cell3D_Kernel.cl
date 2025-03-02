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

constant uint NUM_FACES = 320;
constant uint NUM_VERTICES = 162;

float4 GetCOM(__global float4* Verts ,int ci){
  float4 COM = (float4)(0.0f,0.0f,0.0f,0.0f);
  for(int i = 0; i < NUM_VERTICES; i++){
    COM += Verts[ci * NUM_VERTICES + i];
  }
  return COM/NUM_VERTICES;
}

__kernel void VolumeForceUpdate(__global const uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, int NCELLS, __global float* Kv, __global float* v0){
  uint ci = get_global_id(0);
  uint fi = get_global_id(1);
  uint4 vert_indicies = ci * NUM_VERTICES + VertIdxMat[fi];
  // Get the positions of the vertices of the current face
  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  // Calculate the volume contribution of the current face using the scalar triple product
  float volume = 0.0f;
  for(uint cifi=ci*NUM_VERTICES;cifi<ci*NCELLS+NUM_FACES;cifi++){
    uint4 face = VertIdxMat[cifi];
    float4 vert0  = Verts[face[0]]; 
    float4 vert1  = Verts[face[1]]; 
    float4 vert2  = Verts[face[2]]; 
    float volumepart =  dot(cross(vert1,vert2),vert0);
    if(volumepart < 0){
      volumepart *= -1;
    }
    volume+=volumepart;
  }

  volume /= 6.0f;

  // Calculate the volume strain
  float volumeStrain = (volume / v0[ci]) - 1.0;

  // Initialize force vectors for the vertices
  float4 force0 = (float4)(0.0f);
  float4 force1 = (float4)(0.0f);
  float4 force2 = (float4)(0.0f);

  // Calculate the normal vector of the face
  float4 A = pos1 - pos0;
  float4 B = pos2 - pos0;
  float4 normal = normalize(cross(A, B));

  // Update the forces acting on the vertices based on the volume strain
  Forces[vert_indicies[0]] -= Kv[ci] * 1.0f / 3.0f * volumeStrain * normal;
  Forces[vert_indicies[1]] -= Kv[ci] * 1.0f / 3.0f * volumeStrain * normal;
  Forces[vert_indicies[2]] -= Kv[ci] * 1.0f / 3.0f * volumeStrain * normal;
}

__kernel void SurfaceAreaForceUpdate(__global uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, uint NCELLS, __global float* Ka, __global float* l0){
  uint ci = get_global_id(0);
  uint fi = get_global_id(1);
  uint4 vert_indicies = ci * NUM_VERTICES + VertIdxMat[fi];

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
  float4 dli = (float4)(lengths[0]/l0[ci] - 1.0, lengths[1]/l0[ci] - 1.0, lengths[2]/l0[ci] - 1.0, 0.0);

  // Calculate the force contributions for each vertex
  float third = 1.0 / 3.0;
  Forces[vert_indicies[0]] += third * Ka[ci] * (dli[0] * ulv0 - dli[2] * ulv2);
  Forces[vert_indicies[1]] += third * Ka[ci] * (dli[1] * ulv1 - dli[0] * ulv0);
  Forces[vert_indicies[2]] += third * Ka[ci] * (dli[2] * ulv2 - dli[1] * ulv1);
}

__kernel void StickToSurface(__global uint4* VertIdxMat,__global float4* Verts, __global float4* Forces, uint NCELLS, __global float* Ks, __global float* l0){
  uint ci = get_global_id(0);
  uint fi = get_global_id(1);
  uint4 vert_indicies = ci * NUM_VERTICES + VertIdxMat[fi];

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
    Forces[vert_indicies[0]][2] -= Ks[ci] * (pos0[2]/l0[ci]) * (1.0f/3.0f);
    isunder[0] = 1;
}
  if(pos1[2] < 0.0f){
    Forces[vert_indicies[1]][2] -= Ks[ci] * (pos1[2]/l0[ci]) * (1.0f/3.0f);
    isunder[1] = 1;
  }
  if(pos2[2] < 0.0f){
    Forces[vert_indicies[2]][2] -= Ks[ci] * (pos2[2]/l0[ci]) * (1.0f/3.0f);
    isunder[2] = 1;
  }

  float4 COM = GetCOM(Verts, ci);
  barrier(CLK_LOCAL_MEM_FENCE);

  //if(isnormal && !isunder[0] && pos0[2] < l0[ci]){
  if(isnormal && !isunder[0]){
    Forces[vert_indicies[0]] += (1.0f/3.0f) * Ks[ci] * ((1.0f - pos0[2])/l0[ci]) * normalize(surfacePoint0 - COM);
  }
  if(isnormal && !isunder[1]){
    Forces[vert_indicies[1]] += (1.0f/3.0f) * Ks[ci] * ((1.0f - pos1[2])/l0[ci]) * normalize(surfacePoint1 - COM);
  }
  if(isnormal && !isunder[2]){
    Forces[vert_indicies[2]] += (1.0f/3.0f) * Ks[ci] * ((1.0f - pos2[2])/l0[ci]) * normalize(surfacePoint2 - COM);
  }

  
}

__kernel void RepellingForces(
    __global uint4* VertIdxMat, 
    __global float4* Verts, 
    __global float4* Forces, 
    uint NCELLS, 
    __global float* l0,
     float Kc, 
     int PBC, 
     float L)
  {
  // Get the global IDs for the current cell and face
  uint ci = get_global_id(0);
  uint fi = get_global_id(1);
  uint4 vert_indicies = ci * NUM_VERTICES + VertIdxMat[fi];

  // Get the positions of the vertices of the current face
  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];
  float4 pos0j, pos1j, pos2j;

  // Calculate the center of mass (COM) for the current cell
  float4 COM = GetCOM(Verts, ci);
  uint4 FaceIndexCI = VertIdxMat[fi];

  // Calculate the center of mass for the current face
  float4 faceCOM = (pos0 + pos1 + pos2) / 3.0f;
  float4 faceCOMj;

  // Calculate the normal vector for the current face
  float4 normali = normalize(cross(pos1 - pos0, pos2 - pos0));
  float4 normalj;

  barrier(CLK_LOCAL_MEM_FENCE);
  // Loop over all other cells to calculate repelling forces
  for(uint cj = 0; cj < NCELLS; cj++){
    // Skip the current cell
    if(cj == ci){
      continue;
    }
    // Loop over all faces of the other cell
    for(uint fj = 0; fj < NUM_FACES; fj++){
      // Calculate the face index and get the vertex indices for the other face
      uint4 vert_indicies_j = cj * NUM_VERTICES + VertIdxMat[fj];

      // Get the positions of the vertices of the other face
      pos0j = Verts[vert_indicies_j[0]];
      pos1j = Verts[vert_indicies_j[1]];
      pos2j = Verts[vert_indicies_j[2]];

      // Calculate the center of mass for the other face
      faceCOMj = (pos0j + pos1j + pos2j) / 3.0f;
      // Calculate the normal vector for the other face
      normalj = normalize(cross(pos1j - pos0j, pos2j - pos0j));

      // Calculate the distance vector between the two face centers of mass
      float4 d = faceCOM - faceCOMj;
      // Apply periodic boundary conditions if necessary
      if(PBC){
        d -= L * round(d / L);
      }
      // Calculate the distance between the two face centers of mass
      float dist = sqrt(dot(d,d));
      // Skip if the faces have not crossed eachother or if the distance is greater than the cutoff distance
      if(dot(d, normali) < 0.0f || dist > l0[ci]){
        continue;
      }

      // Calculate the repelling force and update the forces acting on the vertices
      float4 force = Kc * normalize(COM - faceCOM) * (dist/l0[ci]);
      Forces[vert_indicies[0]] += force;
      Forces[vert_indicies[1]] += force;
      Forces[vert_indicies[2]] += force;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

}

__kernel void EulerPosition(__global float4* Verts, __global float4* Forces ,float dt){
  uint ci = get_global_id(0);
  uint vi = get_global_id(1);


  uint vert_index = ci * NUM_VERTICES + vi;
  Verts[vert_index] += Forces[vert_index] * dt;
  Forces[vert_index] = (float4)(0.0f);
}
