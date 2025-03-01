#include "Tissue.hpp"
#include "cell.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <thread>
#include <CL/cl.h>
#include <array>
#include <CL/opencl.hpp>
#include <stdexcept>
#include <fstream>
#include <string>
#include <iostream>

//need to make this global function
std::string readKernelSource(const std::string& filename){
  std::ifstream file(filename);
  if(!file.is_open()){
    throw std::runtime_error("Failed to  find kernel file: " + filename);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}


namespace DPM{
  Tissue3D::Tissue3D(std::vector<DPM::Cell3D> cells,float phi0){
    Cells = cells;
    NCELLS  = Cells.size();
    float volume = 0.0f;
    for(int ci=0;ci<NCELLS;ci++){
      volume += Cells[ci].v0;
    }
    L = cbrt(volume)/phi0;
    PBC = true;
    attractionMethod.assign("General");
  }

  void Tissue3D::Disperse2D(){
    std::vector<float> X,Y,Fx,Fy;
    X.resize(NCELLS); Y.resize(NCELLS);
    Fx.resize(NCELLS); Fy.resize(NCELLS);
    float ri,rj,yi,yj,xi,xj,dx,dy,dist;
    float ux,uy,ftmp,fx,fy;
    int  i,j,count;
    for(i=0;i<NCELLS;i++){
        X[i] = drand48() * L;
        Y[i] = drand48() * L;
    }
    float oldU = 100, dU = 100;
    count = 0;
    while(dU > 1e-6){
        float  U = 0;
        for(i=0;i<NCELLS;i++){
            Fx[i] = 0.0;
            Fy[i] = 0.0;
        }

        for(i=0;i<NCELLS;i++){
            xi = X[i];
            yi = Y[i];
            ri = Cells[i].r0*2;
            for(j=0;j<NCELLS;j++){
                if(j != i){
                    xj = X[j];
                    yj = Y[j];
                    rj = Cells[j].r0*2;
                    dx = xj-xi;
                    dx -= L*round(dx/L);
                    dy = yj-yi;
                    dy -= L*round(dy/L);
                    dist = sqrt(dx*dx + dy*dy);
                    if(dist < 0.0)
                        dist *= -1;
                    if(dist <= (ri+rj)){
                        ux = dx/dist;
                        uy = dy/dist;
                        ftmp = (1.0-dist/(ri+rj))/(ri+rj);
                        fx = ftmp*ux;
                        fy = ftmp*uy;
                        Fx[i] -= fx;
                        Fy[i] -= fy;
                        Fy[j] += fy;
                        Fx[j] += fx;
                        U += 0.5*(1-(dist/(ri+rj))*(1-dist/(ri+rj)));
                    }
                }
            }
        }
        for(int i=0; i<NCELLS;i++){
            X[i] += 0.01*Fx[i];
            Y[i] += 0.01*Fy[i];
        }
        dU = U-oldU;
        if(dU < 0.0)
            dU *= -1;
        oldU = U;
        count++;
        if(count > 1e5){
            std::cerr << "Warning: Max timesteps for dispersion reached"  << std::endl;
            break;
        }
    }
    for(i=0; i<(int)NCELLS; i++){
      std::array<float,3> com = Cells[i].GetCOM();
        for(j=0;j<(int)Cells[i].NV;j++){
            Cells[i].Verts[j][0] -= com[0];
            Cells[i].Verts[j][1] -= com[1];
            Cells[i].Verts[j][0] -= X[i];
            Cells[i].Verts[j][1] -= Y[i];
        }
    }
  }

  void Tissue3D::CLEulerUpdate(int nsteps, float dt){
    static int NF = Cells[0].NF;
    static int NV = Cells[0].NV;
    std::string kernelSource  = readKernelSource("./src/Cell3D_Kernel.comp");

    // OpenCL Setup
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context({device});

    // Compile the kernel
    cl::Program program(context, kernelSource);

    cl_int err = program.build({device},"-cl-opt-disable -Werror");
    if(err != CL_SUCCESS){
      std::cerr <<"ERR: "  <<  CL_SUCCESS  << " : " << "kernel compilation failed:\n";
      std::string version = device.getInfo<CL_DEVICE_VERSION>();
      std::cerr << "OpenCL version:" << version << "\n";
      std::cerr << "platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
      std::cerr << "device:  " << device.getInfo<CL_DEVICE_NAME>() << "\n";
      std::cerr << "driver version: " << device.getInfo<CL_DRIVER_VERSION>() << "\n";
      std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
      exit(0);
    }

    std::vector<std::array<unsigned int,4>> allFaces;
    allFaces.resize(NCELLS*NF);
    std::vector<std::array<float,4>> allVerts;
    allVerts.resize(NCELLS*NV);
    std::vector<std::array<float,4>> allForces;
    allForces.resize(NCELLS*NV);
    std::vector<float> Kv;
    std::vector<float> Ka;
    std::vector<float> Ks;
    std::vector<float> v0;
    std::vector<float> a0;
    std::vector<float> l0;

    for(int ci=0; ci < NCELLS; ci++){
      Kv.push_back(Cells[ci].Kv);
      Ka.push_back(Cells[ci].Ka);
      Ks.push_back(Cells[ci].Ks);
      v0.push_back(Cells[ci].v0);
      a0.push_back(Cells[ci].a0);
      l0.push_back(sqrt((Cells[ci].a0*4.0f)/sqrt(3.0f)));
      unsigned int fistart = NF * NCELLS;
      unsigned int vistart = NV * NCELLS;
      for(unsigned int vi=vistart;vi<vistart+NV;vi++){
        allVerts[vi] = Cells[ci].Verts[vi-vistart];
        allForces[vi] = Cells[ci].Forces[vi-vistart];
      }
      for(unsigned int fi=vistart;fi<fistart+NV;fi++){
        allVerts[fi][0] = Cells[ci].Faces[fi-fistart][0];
        allVerts[fi][1] = Cells[ci].Faces[fi-fistart][1];
        allVerts[fi][2] = Cells[ci].Faces[fi-fistart][2];
        allVerts[fi][3] = 0.0f;
      }
    }

    cl::Buffer gpuFaces(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(std::array<unsigned int,4>) * NCELLS * NF, allFaces.data());
    cl::Buffer gpuVerts(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NV, allVerts.data());
    cl::Buffer gpuForces(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NV, allForces.data());
    cl::Buffer gpuKv(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), Kv.data());
    cl::Buffer gpuKa(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), Ka.data());
    cl::Buffer gpuKs(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), Ks.data());
    cl::Buffer gpuv0(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), v0.data());
    cl::Buffer gpua0(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), a0.data());
    cl::Buffer gpul0(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), l0.data());

    cl::Kernel VolumeUpdateKernel(program,"VolumeForceUpdate");
    VolumeUpdateKernel.setArg(0, gpuFaces);
    VolumeUpdateKernel.setArg(1, gpuVerts);
    VolumeUpdateKernel.setArg(2, gpuForces);
    VolumeUpdateKernel.setArg(3, NCELLS);
    VolumeUpdateKernel.setArg(4, gpuKv);
    VolumeUpdateKernel.setArg(5, gpuv0);

    cl::Kernel SurfaceAreaUpdateKernel(program,"SurfaceAreaForceUpdate");
    SurfaceAreaUpdateKernel.setArg(0, gpuFaces);
    SurfaceAreaUpdateKernel.setArg(1, gpuVerts);
    SurfaceAreaUpdateKernel.setArg(2, gpuForces);
    SurfaceAreaUpdateKernel.setArg(3, NCELLS);
    SurfaceAreaUpdateKernel.setArg(4, gpuKa);
    SurfaceAreaUpdateKernel.setArg(5, gpul0);

    cl::Kernel StickToSurfaceUpdate(program,"StickToSurface");
    StickToSurfaceUpdate.setArg(0, gpuFaces);
    StickToSurfaceUpdate.setArg(1, gpuVerts);
    StickToSurfaceUpdate.setArg(2, gpuForces);
    StickToSurfaceUpdate.setArg(3, NCELLS);
    StickToSurfaceUpdate.setArg(4, gpuKs);
    StickToSurfaceUpdate.setArg(5, gpua0);
    StickToSurfaceUpdate.setArg(6, gpul0);

    cl::Kernel EulerUpdate(program,"EulerPosition");
    EulerUpdate.setArg(0, gpuVerts);
    EulerUpdate.setArg(1, gpuForces);
    EulerUpdate.setArg(2, NCELLS);
    EulerUpdate.setArg(3, dt);

    cl::NDRange globalSize(NCELLS,NF);
    cl::CommandQueue queue(context,device);

    for(int step=0;step<nsteps;step++){
      queue.enqueueNDRangeKernel(VolumeUpdateKernel,cl::NullRange, globalSize);
      queue.enqueueNDRangeKernel(SurfaceAreaUpdateKernel,cl::NullRange, globalSize);
      queue.enqueueNDRangeKernel(StickToSurfaceUpdate,cl::NullRange, globalSize);
      if(step == nsteps-1){
        queue.enqueueReadBuffer(gpuForces, CL_TRUE, 0, sizeof(std::array<float,4>) * NV * NCELLS, allForces.data());
      }
      queue.enqueueNDRangeKernel(EulerUpdate,cl::NullRange, cl::NDRange(NCELLS,NV));
    }

    //add step to place data back into cells


    for(int ci=0;ci<NCELLS;ci++){
      Cells[ci].Volume = Cells[ci].GetVolume();
      Cells[ci].SurfaceArea = Cells[ci].GetSurfaceArea();
    }
    queue.enqueueReadBuffer(gpuVerts, CL_TRUE, 0, sizeof(std::array<float,4>) * NV * NCELLS, allVerts.data());
  }
}
