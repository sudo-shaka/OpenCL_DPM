#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cmath>
#include <vector>
#include <array>
#include <CL/opencl.hpp>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include "cell.hpp"

std::string readKernelSource(const std::string& filename){
  std::ifstream file(filename);
  if(!file.is_open()){
    throw std::runtime_error("Failed to  find kernel file: " + filename);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

namespace DPM{
  Cell2D::Cell2D(float x0, float y0, float CalA, int numVerts, float r0){

  }

  Cell3D::Cell3D(std::array<float,3> starting_point,float  calA, float _r0){
    calA0 = calA;
    r0 = _r0;
    Kv = 0.0f;
    Ka = 0.0f;
    Verts.resize(12);
    float t = (1+sqrt(5)) / 2; //only need first 3, 0 at end is padded for GPU float4
    Verts[0] = {-1,  t, 0, 0};
    Verts[0] = {-1,  t, 0, 0};
    Verts[1] = { 1,  t, 0, 0};
    Verts[2] = {-1, -t, 0, 0};
    Verts[3] = { 1, -t, 0, 0};

    Verts[4] = { 0, -1,  t, 0};
    Verts[5] = { 0,  1,  t, 0};
    Verts[6] = { 0, -1, -t, 0};
    Verts[7] = { 0,  1, -t, 0};

    Verts[8] = { t,  0, -1, 0};
    Verts[9] = { t,  0,  1, 0};
    Verts[10] = {-t,  0, -1, 0};
    Verts[11] = {-t,  0,  1, 0};
    for(int i=0;i<12;i++){
      float norm = std::sqrt(Verts[i][0] * Verts[i][0] + Verts[i][1] * Verts[i][1] + Verts[i][2] * Verts[i][2]);
      Verts[i][0] /= norm;
      Verts[i][1] /= norm;
      Verts[i][2] /= norm;
    }
    Faces.push_back(std::array<int,4>{0,11,5,0});
    Faces.push_back(std::array<int,4>{0,5,1,0});
    Faces.push_back(std::array<int,4>{0, 1, 7,0});
    Faces.push_back(std::array<int,4>{0, 7, 10, 0});
    Faces.push_back(std::array<int,4>{0, 10, 11,0});

    // 5 adjacent faces
    Faces.push_back(std::array<int,4>{1, 5, 9, 0});
    Faces.push_back(std::array<int,4>{5, 11, 4, 0});
    Faces.push_back(std::array<int,4>{11, 10, 2, 0});
    Faces.push_back(std::array<int,4>{10, 7, 6, 0});
    Faces.push_back(std::array<int,4>{7, 1, 8, 0});

    // 5 faces around point 3
    Faces.push_back(std::array<int,4>{3, 9, 4, 0});
    Faces.push_back(std::array<int,4>{3, 4, 2, 0});
    Faces.push_back(std::array<int,4>{3, 2, 6, 0});
    Faces.push_back(std::array<int,4>{3, 6, 8, 0});
    Faces.push_back(std::array<int,4>{3, 8, 9, 0});

    // 5 adjacent faces
    Faces.push_back(std::array<int,4>{4, 9, 5, 0});
    Faces.push_back(std::array<int,4>{2, 4, 11, 0});
    Faces.push_back(std::array<int,4>{6, 2, 10, 0});
    Faces.push_back(std::array<int,4>{8, 6, 7, 0});
    Faces.push_back(std::array<int,4>{9, 8, 1, 0});

    std::array<int,4> newF;
    std::vector<std::array<int,4>> newFaces;
    int f = 2;
    for(int i=0; i < f; i++){
      int steps = Faces.size();
      for(int j=0; j < steps; j++){
        int a = Cell3D::AddMiddlePoint(Faces[j][0], Faces[j][1]);
        int b = Cell3D::AddMiddlePoint(Faces[j][1], Faces[j][2]);
        int c = Cell3D::AddMiddlePoint(Faces[j][2], Faces[j][0]);
        newF = {Faces[j][0],a,c,0};
        newFaces.push_back(newF);
        newF = {Faces[j][1],b,a,0};
        newFaces.push_back(newF);
        newF = {Faces[j][2],c,b,0};
        newFaces.push_back(newF);
        newF = {a,b,c,0};
        newFaces.push_back(newF);
      }
      Faces = newFaces;
      newFaces.clear();
      newFaces.shrink_to_fit();
    }
    Forces.resize(NV);
    for(int vi=0;vi<NV;vi++){
      Verts[vi][0] *= r0;
      Verts[vi][1] *= r0;
      Verts[vi][2] *= r0;
      Verts[vi][0] += starting_point[0];
      Verts[vi][1] += starting_point[1];
      Verts[vi][2] += starting_point[2];
      Forces[vi] = {0,0,0,0};
    }
    v0 = (4.0f/3.0f) * M_PI *pow(r0,3);
    sa0 = pow((6*sqrt(M_PI)*v0*calA),(2.0f/3.0f));
    a0 = (sa0/(float)NF);
    Volume = GetVolume();
    SurfaceArea = GetSurfaceArea();
  }

  int Cell3D::AddMiddlePoint(int p1, int p2){
    int key; int i;
    if(p1 < p2){
        key = floor((p1+p2) * (p1+p2+1)/2) + p1;
    }
    else{
        key = floor((p1+p2) * (p1+p2+1)/2) + p2;
    }
    for(i=0;i<(int)midpointCache.size();i++){
        if(key == midpointCache[i][0])
            return midpointCache[i][1];
    }

    std::array<float,4> vert1 = Verts[p2], vert2 = Verts[p1];
    std::array<float,4> middlePoint;
    for(int i=0;i<3;i++){
      middlePoint[i] = vert1[i] + vert2[i];
      middlePoint[i] *= 0.5;
    }
    for(int i=0;i<3;i++){
      float norm = std::sqrt(middlePoint[0] * middlePoint[0]
          + middlePoint[1] * middlePoint[1]
          + middlePoint[2] * middlePoint[2]);
      middlePoint[i] /=  norm;
    }
    middlePoint[3] = 0;
    Verts.push_back(middlePoint);
    i = Verts.size()-1;
    std::vector<int> cache; cache.resize(2);
    cache.shrink_to_fit();
    cache[0] = key; cache[1] = i;
    midpointCache.push_back(cache);
    return i;
  }

  void Cell3D::CLShapeEuler(int nsteps, float dt){
    int NCELLS = 1;
    float l0 = sqrt((4.0*a0)/sqrt(3.0));
    std::string kernelSource  = readKernelSource("/home/shaka/Code/C++/OpenCL_DPM/src/Cell3D_Kernel.cl");

    // OpenCL Setup
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context({device});

    // Compile the kernel
    cl::Program program(context, kernelSource);

    cl_int err = program.build({device},"-cl-opt-disable -Werror");
    if(err != CL_SUCCESS){
      std::cerr <<"kernel compilation failed:\n";
      std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
      exit(0);
    }

    cl::Buffer gpuFaces(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NF, Faces.data());
    cl::Buffer gpuVerts(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NV, Verts.data());
    cl::Buffer gpuForces(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NV, Forces.data());

    cl::Kernel VolumeUpdateKernel(program,"VolumeForceUpdate");
    VolumeUpdateKernel.setArg(0, gpuFaces);
    VolumeUpdateKernel.setArg(1, gpuVerts);
    VolumeUpdateKernel.setArg(2, gpuForces);
    VolumeUpdateKernel.setArg(3, NCELLS);
    VolumeUpdateKernel.setArg(4, Kv);
    VolumeUpdateKernel.setArg(5, v0);

    cl::Kernel SurfaceAreaUpdateKernel(program,"SurfaceAreaForceUpdate");
    SurfaceAreaUpdateKernel.setArg(0, gpuFaces);
    SurfaceAreaUpdateKernel.setArg(1, gpuVerts);
    SurfaceAreaUpdateKernel.setArg(2, gpuForces);
    SurfaceAreaUpdateKernel.setArg(3, NCELLS);
    SurfaceAreaUpdateKernel.setArg(4, Ka);
    SurfaceAreaUpdateKernel.setArg(5, l0);

    cl::Kernel StickToSurfaceUpdate(program,"StickToSurface");
    StickToSurfaceUpdate.setArg(0, gpuFaces);
    StickToSurfaceUpdate.setArg(1, gpuVerts);
    StickToSurfaceUpdate.setArg(2, gpuForces);
    StickToSurfaceUpdate.setArg(3, NCELLS);
    StickToSurfaceUpdate.setArg(4, Ks);
    StickToSurfaceUpdate.setArg(5, a0);
    StickToSurfaceUpdate.setArg(6, l0);

    cl::Kernel EulerUpdate(program,"EulerPosition");
    EulerUpdate.setArg(0, gpuVerts);
    EulerUpdate.setArg(1, gpuForces);
    EulerUpdate.setArg(2, NCELLS);
    EulerUpdate.setArg(3, dt);

    cl::NDRange globalSize(NCELLS,NF);
    cl::CommandQueue queue(context,device);

    for(int step=0;step<nsteps;step++){
      if(Kv != 0.0){
        queue.enqueueNDRangeKernel(VolumeUpdateKernel,cl::NullRange, globalSize);
      }
      if(Ka != 0.0){
        queue.enqueueNDRangeKernel(SurfaceAreaUpdateKernel,cl::NullRange, globalSize);
      }
      if(Ks != 0.0){
        queue.enqueueNDRangeKernel(StickToSurfaceUpdate,cl::NullRange, globalSize);
      }
      if(step == nsteps-1){
        queue.enqueueReadBuffer(gpuForces, CL_TRUE, 0, sizeof(std::array<float,4>) * NV * NCELLS, Forces.data());
      }
      queue.enqueueNDRangeKernel(EulerUpdate,cl::NullRange, cl::NDRange(NCELLS,NV));
    }
    Volume = GetVolume();
    SurfaceArea = GetSurfaceArea();
    queue.enqueueReadBuffer(gpuVerts, CL_TRUE, 0, sizeof(std::array<float,4>) * NV * NCELLS, Verts.data());
  }


  float Cell3D::GetVolume(){
    float volume = 0.0;
    for(const auto& tri : Faces){
      std::array<float,4> v0 = Verts[tri[0]];
      std::array<float,4> v1 = Verts[tri[1]];
      std::array<float,4> v2 = Verts[tri[2]];
      std::array<float,3> cross;
      cross[0] = v1[1] * v2[2] - v1[2] * v2[1];
      cross[1] = v1[2] * v2[0] - v1[0] * v2[2];
      cross[2] = v1[0] * v2[1] - v1[1] * v2[0];
      float partialVolume = cross[0] * v0[0] + cross[1] * v0[1] + cross[2] * v0[2];
      volume += partialVolume;
    }
    return std::abs(volume)/6.0;   
  }

  float Cell3D::GetSurfaceArea(){
    float SurfaceArea = 0.0;
    for(const auto& tri : Faces){
      std::array<float,4> v0 = Verts[tri[0]];
      std::array<float,4> v1 = Verts[tri[1]];
      std::array<float,4> v2 = Verts[tri[2]];
      std::array<float,4> A = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], 0};
      std::array<float,4> B = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2], 0};
      std::array<float,3> cross;
      cross[0] = A[1] * B[2] - A[2] * B[1];
      cross[1] = A[2] * B[0] - A[0] * B[2];
      cross[2] = A[0] * B[1] - A[1] * B[0];
      float partialArea = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
      SurfaceArea += partialArea;
    }
    return SurfaceArea;
  }

  std::array<std::array<float,162>,3> Cell3D::GetPositions(){
    std::array<std::array<float,NV>,3> positions;
    for(int i=0;i<NV;i++){
      positions[0][i] = Verts[i][0];
      positions[1][i] = Verts[i][1];
      positions[2][i] = Verts[i][2];
    }
    return positions;
  }

  std::array<std::array<float,162>,3> Cell3D::GetForces(){
    std::array<std::array<float,NV>,3> forces;
    for(int i=0;i<NV;i++){
      forces[0][i] = Forces[i][0];
      forces[1][i] = Forces[i][1];
      forces[2][i] = Forces[i][2];
    }
    return forces;
  }

}

