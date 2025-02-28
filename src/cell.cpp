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
    Kb = 0.0f;
    NV = 12;
    Verts.resize(NV);
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
    for(int i=0;i<NV;i++){
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
    NV = (int)Verts.size();
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
    NF = (int)Faces.size();
    v0 = (4.0f/3.0f) * M_PI *pow(r0,3);
    s0 = pow((6*sqrt(M_PI)*v0*calA),(2.0f/3.0f));
    a0 = (s0/(float)NF);
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

  void Cell3D::VolumeForceUpdate(){
    int NCELLS = 1;
    std::string kernelSource  = readKernelSource("./src/Cell3D_Kernel.cl");

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
    }

    cl::Buffer gpuFaces(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NF, Faces.data());
    cl::Buffer gpuVerts(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NV, Verts.data());
    cl::Buffer gpuForces(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(std::array<float,4>) * NCELLS * NV, Forces.data());

    cl::Kernel kernel(program,"VolumeForceUpdate");
    kernel.setArg(0, gpuFaces);
    kernel.setArg(1, gpuVerts);
    kernel.setArg(2, gpuForces);
    kernel.setArg(3, NCELLS);
    kernel.setArg(4, v0);
    kernel.setArg(5, Kb);

    cl::NDRange globalSize(NCELLS,NF);
    cl::CommandQueue queue(context,device);

    queue.enqueueNDRangeKernel(kernel,cl::NullRange, globalSize);
    queue.enqueueReadBuffer(gpuVerts, CL_TRUE, 0, sizeof(std::array<float,4>) * NV * NCELLS, Verts.data());


    std::cout << Forces[0][0] << " "<<  Forces[0][1] <<  " " <<Forces[0][2] << std::endl;
  }
}

