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

float getVolume(const __global uint4* VertIdxMat, __global float4* Verts, int ci){
  float volume = 0.0f;
  for(int i = 0; i < NUM_FACES; i++){
    uint4 face = VertIdxMat[i];
    float4 vert0  = Verts[ci*NUM_VERTICES+face[0]];
    float4 vert1  = Verts[ci*NUM_VERTICES+face[1]];
    float4 vert2  = Verts[ci*NUM_VERTICES+face[2]];
    float volumepart = dot(cross(vert1,vert2),vert0);
    volume += volumepart;
  }
  return fabs(volume) / 6.0f;
}

float4 GetCOM(__global float4* Verts ,int ci){
  float4 COM = (float4)(0.0f,0.0f,0.0f,0.0f);
  for(int i = 0; i < NUM_VERTICES; i++){
    COM += Verts[ci * NUM_VERTICES + i];
  }
  return COM/NUM_VERTICES;
}

float crossProduct(float4 p0, float4 p1, float4 p2){
  return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
}

int findReferencePoint(__global float4* Verts){
  int ci = get_global_id(1);
  int refIndex = ci;
  for(int i= ci*NUM_VERTICES; i < (ci+1) * NUM_VERTICES; i++){
    if(Verts[i].y < Verts[refIndex].y || (Verts[i].y == Verts[refIndex].y && Verts[i].x < Verts[refIndex].x)){
      refIndex = i;
    }
  } 
  return refIndex;
}

float polarAngle(float4 p0, float4 p1){
  return atan2(p1.y - p0.y, p1.x - p0.x);
}

int ConvexHullIndexes(__global float4* Verts, int hullIndexes[NUM_VERTICES]){
  int ci = get_global_id(1);
  int refIndex = findReferencePoint(Verts);
  float4 refPoint = Verts[refIndex];
  for(int i = ci*NUM_VERTICES; i < (ci+1) * NUM_VERTICES; i++){
    for(int j=0; j < NUM_VERTICES - i -1;j++){
      if(polarAngle(refPoint,Verts[j])>polarAngle(refPoint,Verts[j+1])){
        float4 temp = Verts[j];
        Verts[j] = Verts[j+1];
        Verts[j+1] = temp;
      }
    }
  }

  int hullSize = NUM_VERTICES;
  hullIndexes[0] = refIndex;
  hullIndexes[1] = ci*NUM_VERTICES + 1;
  hullIndexes[2] = ci*NUM_VERTICES + 2;
  for(int i = (ci * NUM_VERTICES) + 3; i < (ci+1) * NUM_VERTICES; i++){
    while(hullSize > 2 && crossProduct(Verts[hullIndexes[hullSize-2]], Verts[hullIndexes[hullSize-1]], Verts[i] - Verts[i]) <= 0.0f){
      hullSize--;
    }
    hullIndexes[hullSize] = i;
    hullSize++;
  }
  hullIndexes[hullSize] = NULL;
  return hullSize;
}


__kernel void VolumeForceUpdate(__global const uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, int NCELLS, __global float* Kv, __global float* v0){
  uint ci = get_global_id(1);
  uint fi = get_global_id(0);
  uint gi = get_group_id(1);
  uint li = get_local_id(0);


  if(Kv[ci] == 0){
    return;
  }
  uint4 vert_indicies = ci * NUM_VERTICES + VertIdxMat[fi];
  // Get the positions of the vertices of the current face
  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  // Calculate the volume contribution of the current face using the scalar triple product
  __local float localVolume;
  if(fi % 25 == 0){ //for some reason you need to update the local volume every 25 faces (gpu block size?)
    localVolume = getVolume(VertIdxMat, Verts, ci);
  }

  // Calculate the volume strain
  float volumeStrain = (localVolume / v0[ci]) - 1.0;

  // Initialize force vectors for the vertices
  float4 force0 = (float4)(0.0f);
  float4 force1 = (float4)(0.0f);
  float4 force2 = (float4)(0.0f);

  // Calculate the normal vector of the face
  float4 A = pos1 - pos0;
  float4 B = pos2 - pos0;
  float4 normal = normalize(cross(A, B));

  // Update the forces acting on the vertices based on the volume strain
  float third = 1.0f / 3.0f;
  Forces[vert_indicies[0]] -= Kv[ci] * third * volumeStrain * normal;
  Forces[vert_indicies[1]] -= Kv[ci] * third * volumeStrain * normal;
  Forces[vert_indicies[2]] -= Kv[ci] * third * volumeStrain * normal;
  barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void SurfaceAreaForceUpdate(__global uint4* VertIdxMat, __global float4* Verts, __global float4* Forces, uint NCELLS, __global float* Ka, __global float* l0){
  uint ci = get_global_id(1);
  uint fi = get_global_id(0);
  if(Ka[ci] == 0){
    return;
  }
  uint4 vert_indicies = ci * NUM_VERTICES + VertIdxMat[fi];

  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  // Calculate the lengths of the edges of the triangle
  float4 lv0 = pos1 - pos0;
  float4 lv1 = pos2 - pos1;
  float4 lv2 = pos0 - pos2;
  float4 lengths = (float4)(length(lv0), length(lv1), length(lv2), 0.0f);

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
  uint ci = get_global_id(1);
  uint fi = get_global_id(0);
  if(Ks[ci] == 0){
    return;
  }
  uint4 vert_indicies = ci * NUM_VERTICES + VertIdxMat[fi];

  float4 pos0 = Verts[vert_indicies[0]];
  float4 pos1 = Verts[vert_indicies[1]];
  float4 pos2 = Verts[vert_indicies[2]];

  float4 A = pos1 - pos0;
  float4 B = pos2 - pos0;
  bool isnormal = (A.x*B.y - A.y*B.x) < 0.0f;
  if(!isnormal){
    return;
  }


  if(pos0[2] < 0.0f){
    Forces[vert_indicies[0]][2] -= Ks[ci] * (pos0[2]/l0[ci]) * (1.0f/3.0f);
}
  if(pos1[2] < 0.0f){
    Forces[vert_indicies[1]][2] -= Ks[ci] * (pos1[2]/l0[ci]) * (1.0f/3.0f);
  }
  if(pos2[2] < 0.0f){
    Forces[vert_indicies[2]][2] -= Ks[ci] * (pos2[2]/l0[ci]) * (1.0f/3.0f);
  }

  float4 COM = GetCOM(Verts, ci);
  barrier(CLK_LOCAL_MEM_FENCE);

  if(pos0[2] < l0[ci] && pos1[2] < l0[ci] && pos2[2] < l0[ci]){
    Forces[vert_indicies[0]] += (1.0f/3.0f) * Ks[ci] * normalize(pos0 - COM);
    Forces[vert_indicies[1]] += (1.0f/3.0f) * Ks[ci] * normalize(pos1 - COM);
    Forces[vert_indicies[2]] += (1.0f/3.0f) * Ks[ci] * normalize(pos2 - COM);
  }
  /*if(pos1[2] < l0[ci]){
    //Forces[vert_indicies[1]] += (1.0f/3.0f) * Ks[ci] * normalize(pos1 - COM);
  }
  if(pos2[2] < l0[ci]){
    //Forces[vert_indicies[2]] += (1.0f/3.0f) * Ks[ci] * normalize(pos2 - COM);
  }*/
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
  uint ci = get_global_id(1);
  uint fi = get_global_id(0);

  if(Kc == 0){
    return;
  }

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
      //float4 force = Kc * normalize(faceCOM - faceCOMj) * (dist/l0[ci]);
      //float4 force = Kc * normalize(COM-faceCOM) * (dist/l0[ci]);
      Forces[vert_indicies[0]] += Kc * normalize(COM - pos0) * (dist/l0[ci]);
      Forces[vert_indicies[1]] += Kc * normalize(COM - pos1) * (dist/l0[ci]);
      Forces[vert_indicies[2]] += Kc * normalize(COM - pos2) * (dist/l0[ci]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

__kernel void AllVertAVttraction(__global float4* Verts, __global float4* Forces, __global float* l0, float L , int NCELLS, int PBC, float Kat){
  float dist;
  uint ci = get_global_id(1);
  uint vi = get_global_id(0);
  if(Kat == 0){
    return;
  }
  uint vert_index = ci * NUM_VERTICES + vi;
  float4 p1 = Verts[vert_index];
  for(int cj=0;cj<NCELLS;cj++){
    if(ci != cj){
      for(int vj=0;vj<NUM_VERTICES;vj++){
        float4 p2 = Verts[cj * NUM_VERTICES + vj];
        float4 delta = p2-p1;
        if(PBC){
          delta -= L*round(delta/L);
        }
        dist = sqrt(dot(delta,delta));
        if(dist < l0[ci]){
          Forces[vert_index] -= Kat * 0.5f * (dist/l0[ci] - 1.0f) * normalize(delta);
          Forces[cj * NUM_VERTICES + vj] += Kat * 0.5f * (dist/l0[ci] - 1.0f) * normalize(delta);
        }
      }
    }
  }
}

__kernel void JunctionAttraction(__global float4* Verts, __global float4* Forces, __global float* l0, float L , int NCELLS, int PBC, float Kat){
  float dist;
  uint ci = get_global_id(1);
  uint vi = get_global_id(0);
  if(Kat == 0){
    return;
  }
  uint vert_index = ci * NUM_VERTICES + vi;
  int hullIdx[NUM_VERTICES];
  int nHull = ConvexHullIndexes(Verts, hullIdx);
  for(int i=0; i<nHull; i++){
    if (hullIdx[i] == vert_index){
      return;
    }
  }
  
  float4 p1 = Verts[vert_index];
  for(int cj=0;cj<NCELLS;cj++){
    if(ci != cj){
      for(int vj=0;vj<NUM_VERTICES;vj++){
        float4 p2 = Verts[cj * NUM_VERTICES + vj];
        float4 delta = p2-p1;
        if(PBC){
          delta -= L*round(delta/L);
        }
        dist = sqrt(dot(delta,delta));
        if(dist < l0[ci]){
          Forces[vert_index] -= Kat * 0.5f * (dist/l0[ci] - 1.0f) * normalize(delta);
          Forces[cj * NUM_VERTICES + vj] += Kat * 0.5f * (dist/l0[ci] - 1.0f) * normalize(delta);
        }
      }
    }
  }
}

__kernel void EulerPosition(__global float4* Verts, __global float4* Forces ,float dt){
  uint ci = get_global_id(1);
  uint vi = get_global_id(0);


  uint vert_index = ci * NUM_VERTICES + vi;
  Verts[vert_index] += Forces[vert_index] * dt;
  Forces[vert_index] = (float4)(0.0f);
}
