

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

__kernel void AreaForceUpdates(__global float* xvals, __global float* yvals, __global float* Fx, __global float* Fy, int NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV - 1){
        ip1 -= NV;
        ip2 = ip1 + 1;
    }
    if(vi == NV - 2){
        ip2 -= NV;
    }
    if(vi == 0){
        im1 += NV;
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV;
    }

}

__kernel void BendingForceUpdate(int NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV - 1){
        ip1 -= NV;
        ip2 = ip1 + 1;
    }
    if(vi == NV - 2){
        ip2 -= NV;
    }
    if(vi == 0){
        im1 += NV;
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV;
    }

}

__kernel void PerimeterForceUpdate(int NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV - 1){
        ip1 -= NV;
        ip2 = ip1 + 1;
    }
    if(vi == NV - 2){
        ip2 -= NV;
    }
    if(vi == 0){
        im1 += NV;
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV;
    }

}

__kernel void RepulsionForceUpdate(int NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV - 1){
        ip1 -= NV;
        ip2 = ip1 + 1;
    }
    if(vi == NV - 2){
        ip2 -= NV;
    }
    if(vi == 0){
        im1 += NV;
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV;
    }

}

__kernel void AttractionForceUpdate(int NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV - 1){
        ip1 -= NV;
        ip2 = ip1 + 1;
    }
    if(vi == NV - 2){
        ip2 -= NV;
    }
    if(vi == 0){
        im1 += NV;
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV;
    }

}

__kernel void EulerUpdate(int NV){
    int ci = get_global_id(0);
    int vi = get_global_id(1);
    int NCELLS = get_global_size(0);

    int index = ci * NCELLS + vi;
    int ip1 = index+1;
    int ip2 = index+2;
    int im1 = index-1;
    int im2 = index-2;

    if(vi == NV - 1){
        ip1 -= NV;
        ip2 = ip1 + 1;
    }
    if(vi == NV - 2){
        ip2 -= NV;
    }
    if(vi == 0){
        im1 += NV;
        im2 = im1 - 1;
    }
    if(vi == 1){
        im2 += NV;
    }

}