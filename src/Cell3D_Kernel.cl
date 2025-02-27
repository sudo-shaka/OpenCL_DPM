

__kernel void VolumeForceUpdate3D(__global const uint4* VertIdxMat, __global* float4 Verts, __local float*  v0, __local  float* volumes,  int NV,  int NF,  int NCELLS){


  /*data is structred like
      cell1, cell2, cell3, cell4, ...
    f1
    f2
    f3 
    ...
  */
  
  int ci  = get_global_id(0); //cell index
  int fi  = get_global_id(1); //face index at that cell

  float volume = volumes[ci]; 
  float idealVolume = v0[ci];

  //logic is... for each  face in tissue, do volume force update...

  int face_index = ci * NCELLS + fi;
  uint4 vert_indicies = VertIdxMat[face_index];

  float4 pos1 = Verts[vert_indicies[0]];
  float4 pos2 = Verts[vert_indicies[1]];
  float4 pos3 = Verts[vert_indicies[2]];

  float volumeStrain = (volume/v0) - 1.0;
}
