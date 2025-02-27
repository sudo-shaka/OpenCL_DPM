//#define CL_HPP_ENABLE_EXCEPTIONS
//#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include"cell.hpp"

std::string readKernelSource(const std::string& filename){
  std::ifstream file(filename);
  if(!file.is_open()){
    throw std::runtime_error("Failed to  find kernel file: " + filename);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main(){
  std::array<float,3> startp = {0,0,0};
  DPM::Cell3D Cell(startp, 1.05, 1.0);

  for(int vi=0;vi<Cell.NV;vi++){
    std::cout << Cell.Verts[vi][0] << "," << Cell.Verts[vi][1] << ","<< Cell.Verts[vi][2] << std::endl;
  }
  std::string kernelSource  = readKernelSource("./src/matrix_mul.cl");
  // Initialize Matrices
  int N = 4;
  std::vector<float> A(N * N, 1.0f);
  std::vector<float> B(N * N, 2.0f);
  std::vector<float> C(N * N, 0.0f);

  // OpenCL Setup
  cl::Platform platform = cl::Platform::getDefault();
  cl::Device device = cl::Device::getDefault();
  cl::Context context({device});

  // Compile the kernel
  cl::Program program(context, kernelSource);
  program.build({device});

  // Buffers
  cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, A.data());
  cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, B.data());
  cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);

  // Create Kernel
  cl::Kernel kernel(program, "matMul");
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
  }
}
