

float2 GetCOMX(__global float2* positions, int NV){
    int ci = get_global_id(0);
    float2 com = (float2) 0.0f;
    for(int i = 0; i < NV; i++){
        com += positions[ci * NV + i];
    }
    return com / (float)NV;
}

__kernel void AreaForceUpdates(__global float2* Verts, __global float2* Forces, __global int* NV, __global float* a0, __global float* Ka){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NUM_VERTS = get_global_size(1);
    int NCELLS = get_global_size(0);

    if(vi > NV[ci]){
      return;
    }

    int index = ci * NUM_VERTS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV[ci] - 1){
        ip1 -= NV[ci];
        ip2 = ip1 + 1;
    }
    if(vi == NV[ci] - 2){
        ip2 -= NV[ci];
    }
    if(vi == 0){
        im1 += NV[ci];
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV[ci];
    }

    float partialArea = 0.0f;
    float areaStrain = 0.0f;
    float Area;
    for(int vi = 0; vi < NV[ci]; vi++){
        partialArea = 0.5 * ((Verts[im1].x + Verts[index].x) * (Verts[im1].y - Verts[index].y));
        Area += partialArea;
    }
    if(Area < 0.0){
        Area = -Area;
    }
    areaStrain = (Area/a0[ci]) - 1.0;
    Forces[index].x = Ka[index] * 0.5 * areaStrain * (Verts[im1].y - Verts[ip1].y);
    Forces[index].y = Ka[index] * 0.5 * areaStrain * (Verts[im1].x - Verts[ip1].x);
}

__kernel void BendingForceUpdates(__global float2* Verts, __global float2* Forces,__global int* NV, __global float* Kb, __global float* a0, __global float* l0){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NUM_VERTS = get_global_size(1);
    int index = ci * NUM_VERTS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV[ci] - 1){
        ip1 -= NV[ci];
        ip2 = ip1 + 1;
    }
    if(vi == NV[ci] - 2){
        ip2 -= NV[ci];
    }
    if(vi == 0){
        im1 += NV[ci];
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV[ci];
    }
    float2 lv = Verts[ip1] - Verts[index];
    float2 lvm = Verts[index] - Verts[im1];
    float rho0 = sqrt(a0[ci]);
    float fb = Kb[ci]*(rho0/l0[ci]);
    float2 si, sip, sim;
    si = lv - lvm;
    sip = Verts[ip2] - Verts[ip1] - lv;
    sim = lvm - Verts[im1] - Verts[im2];
    Forces[index] += fb * (2.0f*si - sim - sip);
}

__kernel void PerimeterForceUpdates(__global float2* Verts, __global float2* Forces,__global int* NV, __global float* Kl, __global float* a0, __global float* l0){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NUM_VERTS = get_global_size(1);
    int NCELLS = get_global_size(0);

    int index = ci * NUM_VERTS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV[ci] - 1){
        ip1 -= NV[ci];
        ip2 = ip1 + 1;
    }
    if(vi == NV[ci] - 2){
        ip2 -= NV[ci];
    }
    if(vi == 0){
        im1 += NV[ci];
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV[ci];
    }
    float2 lv, lvm, ulv, ulvm;
    float dlim1,dli;
    lv = Verts[ip1] - Verts[index];
    lvm = Verts[index] - Verts[im1];
    float length = distance(Verts[ip1],Verts[index]);
    float lengthm = distance(Verts[index],Verts[im1]);
    ulv = lv / length;
    ulvm = lvm / lengthm;
    dli = length/l0[ci] - 1.0;
    dlim1 = lengthm/l0[ci] - 1.0;
    Forces[index] += Kl[ci] * sqrt(a0[ci]/l0[ci]) * (dli*ulv - dlim1 * ulvm);
}

__kernel void RepulsionForceUpdate(__global int* NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NUM_VERTS = get_global_size(1);

    int index = ci * NUM_VERTS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV[ci] - 1){
        ip1 -= NV[ci];
        ip2 = ip1 + 1;
    }
    if(vi == NV[ci] - 2){
        ip2 -= NV[ci];
    }
    if(vi == 0){
        im1 += NV[ci];
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV[ci];
    }

}

__kernel void AttractionForceUpdate(__global int* NV){
    int ci = get_global_id(0);
    int NUM_VERTS = get_global_size(1);
    int vi = get_global_id(1);

    int index = ci * NUM_VERTS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV[ci] - 1){
        ip1 -= NV[ci];
        ip2 = ip1 + 1;
    }
    if(vi == NV[ci] - 2){
        ip2 -= NV[ci];
    }
    if(vi == 0){
        im1 += NV[ci];
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV[ci];
    }
}

__kernel void EulerUpdate(__global float2* Verts, __global float2* Forces, __global int* NV, float dt){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NUM_VERTS = get_global_size(1);
    if(vi > NV[ci]){
      return;
    }
    int index = ci * NUM_VERTS + vi;
    Verts[index] += Forces[index]*dt;
}
