#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <cmath>
#include <vector>
#include <array>
#include <CL/opencl.hpp>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>
#include "cell.hpp"

#define NV3D = 162

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
    float t = (1+sqrt(5)) / 2;
    Verts[0] = {-1,t,0};
    Verts[0] = {-1, t, 0};
    Verts[1] = { 1,  t,  0};
    Verts[2] = {-1, -t,  0};
    Verts[3] = { 1, -t,  0};

    Verts[4] = { 0, -1,  t};
    Verts[5] = { 0,  1,  t};
    Verts[6] = { 0, -1, -t};
    Verts[7] = { 0,  1, -t};

    Verts[8] = { t,  0, -1};
    Verts[9] = { t,  0,  1};
    Verts[10] = {-t,  0, -1};
    Verts[11] = {-t,  0,  1};
    for(int i=0;i<NV;i++){
      float norm = std::sqrt(Verts[i][0] * Verts[i][0] + Verts[i][1] * Verts[i][1] + Verts[i][2] * Verts[i][2]);
      Verts[i][0] /= norm;
      Verts[i][1] /= norm;
      Verts[i][2] /= norm;
    }
    Faces.push_back(std::array<int,3>{0,11,5});
    Faces.push_back(std::array<int,3>{0,5,1});
    Faces.push_back(std::array<int,3>{0, 1, 7});
    Faces.push_back(std::array<int,3>{0, 7, 10});
    Faces.push_back(std::array<int,3>{0, 10, 11});

    // 5 adjacent faces
    Faces.push_back(std::array<int,3>{1, 5, 9});
    Faces.push_back(std::array<int,3>{5, 11, 4});
    Faces.push_back(std::array<int,3>{11, 10, 2});
    Faces.push_back(std::array<int,3>{10, 7, 6});
    Faces.push_back(std::array<int,3>{7, 1, 8});

    // 5 faces around point 3
    Faces.push_back(std::array<int,3>{3, 9, 4});
    Faces.push_back(std::array<int,3>{3, 4, 2});
    Faces.push_back(std::array<int,3>{3, 2, 6});
    Faces.push_back(std::array<int,3>{3, 6, 8});
    Faces.push_back(std::array<int,3>{3, 8, 9});

    // 5 adjacent faces
    Faces.push_back(std::array<int,3>{4, 9, 5});
    Faces.push_back(std::array<int,3>{2, 4, 11});
    Faces.push_back(std::array<int,3>{6, 2, 10});
    Faces.push_back(std::array<int,3>{8, 6, 7});
    Faces.push_back(std::array<int,3>{9, 8, 1});

    std::array<int,3> newF;
    std::vector<std::array<int,3>> newFaces;
    int f = 2;
    for(int i=0; i < f; i++){
      int steps = Faces.size();
      for(int j=0; j < steps; j++){
        int a = Cell3D::AddMiddlePoint(Faces[j][0], Faces[j][1]); 
        int b = Cell3D::AddMiddlePoint(Faces[j][1], Faces[j][2]); 
        int c = Cell3D::AddMiddlePoint(Faces[j][2], Faces[j][0]); 
        newF = {Faces[j][0],a,c};
        newFaces.push_back(newF);
        newF = {Faces[j][1],b,a};
        newFaces.push_back(newF);
        newF = {Faces[j][2],c,b};
        newFaces.push_back(newF);
        newF = {a,b,c};
        newFaces.push_back(newF);
      }
      Faces = newFaces;
      newFaces.clear();
      newFaces.shrink_to_fit();
    }
    NV = (int)Verts.size();
    Forces.resize(NV);
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

    std::array<float,3> vert1 = Verts[p2], vert2 = Verts[p1];
    std::array<float,3> middlePoint;
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
    Verts.push_back(middlePoint);
    i = Verts.size()-1;
    std::vector<int> cache; cache.resize(2);
    cache.shrink_to_fit();
    cache[0] = key; cache[1] = i;
    midpointCache.push_back(cache);
    return i;
  }

  void Cell3D::VolumeForceUpdate(){
    std::string kernelSource  = readKernelSource("./src/Cell3D_Kernel.cl");

    // OpenCL Setup
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context({device});

    // Compile the kernel
    cl::Program program(context, kernelSource);
    program.build({device});



    // Buffers
/*    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, A.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, B.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);

    // Create Kernel
    cl::Kernel kernel(program, "VolumeForceUpdate3D");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);

    // Command Queue
    cl::CommandQueue queue(context, device);
    cl::NDRange globalSize(N, N);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * N * N, C.data());

    // Print Result
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }*/
  }
}

