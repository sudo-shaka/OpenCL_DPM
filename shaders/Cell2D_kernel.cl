

float GetCOMX(__global float* xvals, int ci ,int NV){
    float COMX = 0.0f;
    for(int i = 0; i < NV; i++){
        COMX += xvals[ci * NV + i];
    }
    return COMX / (float)NV;
}

float GetCOMY(__global float* yvals, int ci ,int NV){
    float COMY = 0.0f;
    for(int i = 0; i < NV; i++){
        COMY += yvals[ci * NV + i];
    }
    return COMY / (float)NV;
}

__kernel void AreaForceUpdates(__global float* xvals, __global float* yvals, __global float* Fx, __global float* Fy, __global int* NV, __global float* a0, __global float* Ka){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
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
        partialArea = 0.5 * ((xvals[im1] + xvals[index]) * (yvals[im1] - yvals[index]));
        Area += partialArea;
    }
    if(Area < 0.0){
        Area = -Area;
    }
    areaStrain = (Area/a0[ci]) - 1.0;
    Fx[index] = Ka[index] * 0.5 * areaStrain * (yvals[im1] - yvals[ip1]);
    Fy[index] = Ka[index] * 0.5 * areaStrain * (xvals[im1] - xvals[ip1]);
}

__kernel void BendingForceUpdate(__global float* xvals, __global float* yvals, __global float* Fx, __global float* Fy,__global int* NV, __global float* Kb, __global float* a0, __global float* l0){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);
    int index = ci * NCELLS + vi;
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
    float lvx = xvals[ip1] - xvals[index];
    float lvy = yvals[ip1] - yvals[index];
    float lvxm = xvals[index] - xvals[im1];
    float lvym = yvals[index] - yvals[im1];
    float rho0 = sqrt(a0[ci]);
    float fb = Kb[ci]*(rho0/l0[ci]);
    float six, sixp, sixm;
    float siy, siyp, siym;
    six = lvx - lvxm;
    siy = lvy - lvym;
    sixp = xvals[ip2] - xvals[ip1] - lvx;
    siyp = yvals[ip2] - yvals[ip1] - lvx;
    sixm = lvxm - xvals[im1] - xvals[im2];
    siym = lvym - yvals[im1] - yvals[im2];
    Fx[index] += fb * (2.0*six - sixm - sixp);
    Fy[index] += fb * (2.0*siy - siym - siyp);
}

__kernel void PerimeterForceUpdate(__global float* xvals, __global float* yvals, __global float* Fx, __global float* Fy,__global int* NV, __global float* Kl, __global float* a0, __global float* l0){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
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

    float lvxm, lvym, lvx, lvy;
    float ulvxm, ulvym, ulxm, ulym, ulvx, ulvy;
    float dlim1,dli;
    lvx = xvals[ip1] - xvals[index];
    lvy = yvals[ip1] - yvals[index];
    lvxm = xvals[index] - xvals[im1];
    lvym = yvals[index] - yvals[im1];
    float length = sqrt(lvx*lvx + lvym*lvym);
    float lengthm = sqrt(lvxm*lvxm + lvym*lvym);
    ulvx = lvx / length;
    ulvy = lvy / length;
    ulvxm = lvxm / lengthm;
    ulvym = lvym / lengthm;
    dli = length/l0[ci] - 1.0;
    dlim1 = lengthm/l0[ci] - 1.0;
    Fx[index] = Kl[ci] * ((sqrt(a0[ci])/l0[ci]) * (dli * ulvx - dlim1 * ulvxm));
    Fy[index] = Kl[ci] * ((sqrt(a0[ci])/l0[ci]) * (dli * ulvy - dlim1 * ulvym));

}

__kernel void RepulsionForceUpdate(__global int* NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
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
    int NCELLS = get_global_size(0);
    int vi = get_global_id(1);
    

    int index = ci * NCELLS + vi;
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

__kernel void EulerUpdate(__global float* xvals, __global float* yvals, __global float* Fx, __global float* Fy, __global int* NV, float dt){
    int ci = get_global_id(0);
    int vi = get_global_id(1);

    int index = ci * NV[ci] + vi;

    xvals[index] += Fx[index] * dt;
    yvals[index] += Fy[index] * dt;
}