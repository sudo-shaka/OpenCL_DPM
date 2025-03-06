

float2 GetCOM(__global float2* positions,int NV){
    int ci = get_global_id(0);
    int NUM_VERTS = get_global_size(1);
    float2 com = (float2) 0.0f;
    for(int i = 0; i < NV; i++){
        com += positions[ci * NUM_VERTS + i];
    }
    return com / (float)NV;
}

__kernel void AreaForceUpdates(__global float2* Verts, __global float2* Forces, __global int* NV, __global float* a0, __global float* Ka){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NUM_VERTS = get_global_size(1);
    int NCELLS = get_global_size(0);

    int index = ci * NUM_VERTS + vi;
    int im1 = (vi == 0) ? (ci * NUM_VERTS + NV[ci] - 1) : (index-1);
    int ip1 = (vi == NV[ci] -1) ? (ci * NUM_VERTS) : (index + 1);

    float Area = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int vj = 0; vj < NV[ci]; vj++){
        int idx = ci * NUM_VERTS + vj;
        int idx_im1 = (vj == 0) ? (ci * NUM_VERTS + NV[ci] - 1) : (idx - 1);
        Area += 0.5 * ((Verts[idx_im1].x + Verts[idx].x) * (Verts[idx_im1].y - Verts[idx].y));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(Area < 0.0){
        Area = -Area;
    }
    float areaStrain = (Area/a0[ci]) - 1.0;
    Forces[index].x += (Ka[ci]/sqrt(a0[ci])) * 0.5 * areaStrain * (Verts[im1].y - Verts[ip1].y);
    Forces[index].y += (Ka[ci]/sqrt(a0[ci])) * 0.5 * areaStrain * (Verts[im1].x - Verts[ip1].x);
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
        ip1 = ci * NUM_VERTS;
        ip2 = ip1 + 1;
    }
    if(vi == NV[ci] - 2){
        ip2 = ci * NUM_VERTS;
    }
    if(vi == 0){
        im1 = ci * NUM_VERTS + NV[ci]-1;
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 = ci* NUM_VERTS + NV[ci]-1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the differences in vertex positions
    float lvx = Verts[ip1].x - Verts[index].x;
    float lvy = Verts[ip1].y - Verts[index].y;
    float lvxm = Verts[index].x - Verts[im1].x;
    float lvym = Verts[index].y - Verts[im1].y;

    // Calculate the second differences (curvatures)
    float six = lvx - lvxm;
    float siy = lvy - lvym;
    float sixp = (Verts[ip2].x - Verts[ip1].x) - lvx;
    float siyp = (Verts[ip2].y - Verts[ip1].y) - lvy;
    float sixm = lvxm - (Verts[im1].x - Verts[im2].x);
    float siym = lvym - (Verts[im1].y - Verts[im2].y);

    // Update the forces to prevent bending
    Forces[index].x += Kb[ci] * (2.0 * six - sixm - sixp);
    Forces[index].y += Kb[ci] * (2.0 * siy - siym - siyp);

}

__kernel void PerimeterForceUpdates(__global float2* Verts, __global float2* Forces,__global int* NV, __global float* Kl, __global float* a0, __global float* l0){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NUM_VERTS = get_global_size(1);
    if(vi > NV[ci]){
        return;
    }

    int index = ci * NUM_VERTS + vi;
    int im1 = (vi == 0) ? (ci * NUM_VERTS + NV[ci] - 1) : (index-1);
    int ip1 = (vi == NV[ci] -1) ? (ci * NUM_VERTS) : (index + 1);

    barrier(CLK_LOCAL_MEM_FENCE);
    float2 lv, lvm, ulv, ulvm;
    float dlim1,dli;
    lv = Verts[ip1] - Verts[index];
    lvm = Verts[index] - Verts[im1];
    float length = sqrt(dot(lv,lv));
    float lengthm = sqrt(dot(lvm,lvm));
    ulv = lv / length;
    ulvm = lvm / lengthm;
    dli = length/l0[ci] - 1.0;
    dlim1 = lengthm/l0[ci] - 1.0;
    Forces[index] += Kl[ci] * sqrt(a0[ci]/l0[ci]) * (dli*ulv - dlim1 * ulvm);
}

__kernel void RepulsionForceUpdate(__global float2* Verts, __global float2* Forces,__global int* NV,__global float* r0 ,int PBC, float L, float Kre){
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

    // Declare variables
    float rij, xij, ftmp, diff;

    // Check if the vertex index is valid
    if (vi >= NV[ci]) {
        return;
    }

    // Synchronize threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialize overlap flag
    bool overlaps = false;

    // Loop over all cells
    for (int cj = 0; cj < NCELLS; cj++) {
        overlaps = false;
        int i, j;

        // Loop over all vertices of the current cell
        for (i = 0, j = NV[cj] - 1; i < NV[cj]; j = i++) {
            int cj_vj_i = cj * NUM_VERTS + i;
            int cj_vj_j = cj * NUM_VERTS + j;
            float2 di = Verts[index] - Verts[cj_vj_i];
            float2 dj = Verts[index] - Verts[cj_vj_j];

            // Apply periodic boundary conditions if necessary
            if(PBC){
                if (fabs(di.x) > L || fabs(dj.x) > L){
                    di.x -= L * floor(di.x / L);
                    dj.x -= L * floor(dj.x / L);
                }
                if(fabs(di.y) > L || fabs(dj.y) > L){
                    di.y -= L*round(di.y/L);
                    dj.y -= L*round(dj.y/L);
                }
            }

            // Check for overlap with other cells
            if (ci != cj) {
                if ((di.y > 0) != (dj.y > 0) && 
                    (0 < (dj.x - di.x) * (0 - di.y) / (dj.y - di.y) + di.x)) {
                    overlaps = !overlaps;
                }
            }
        }

        // If overlap is found, break the loop
        if (overlaps) {
            break;
        }
    }

    // If overlap is detected, apply repulsion force
    if (overlaps) {
        float2 COM = GetCOM(Verts, NV[ci]);
        float2 d = COM - Verts[index];
        float dist = sqrt(dot(d, d));

        // Apply periodic boundary conditions to distance
        dist -= L * round(dist / L);

        // Calculate repulsion force
        xij = dist / (2 * r0[ci]);
        float ftmp = Kre * (1-xij);

        // Update vertex position
        Forces[index] += 0.5f * ftmp * normalize(d);
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
    Forces[index] = (float2)(0.0f);
}
