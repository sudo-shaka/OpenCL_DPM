#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "Tissue.hpp"
#include "cell.hpp"
#include "readKernel.hpp"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/opencl.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace DPM {
Tissue3D::Tissue3D(std::vector<DPM::Cell3D> cells, float phi0) {
  Cells = cells;
  NCELLS = Cells.size();
  if (NCELLS > 100) {
    std::cerr
        << "Warning: Large number of cells, too many cells may crash program!"
        << std::endl;
  }
  float volume = 0.0f;
  for (int ci = 0; ci < NCELLS; ci++) {
    volume += Cells[ci].v0;
  }
  L = cbrt(volume) / phi0;
  PBC = true;
  attractionMethod.assign("General");

  platform = cl::Platform::getDefault();
  device = cl::Device::getDefault();
  context = cl::Context({device});
  // Compile the kernel
  std::string kernelSource = readKernelSource("shaders/Cell3D_Kernel.cl");
  program = cl::Program(context, kernelSource);
}

void Tissue3D::Disperse2D() {
  std::vector<float> X, Y, Fx, Fy;
  X.resize(NCELLS);
  Y.resize(NCELLS);
  Fx.resize(NCELLS);
  Fy.resize(NCELLS);
  float ri, rj, yi, yj, xi, xj, dx, dy, dist;
  float ux, uy, ftmp, fx, fy;
  int i, j, count;
  for (i = 0; i < NCELLS; i++) {
    X[i] = drand48() * L;
    Y[i] = drand48() * L;
  }
  float oldU = 100, dU = 100;
  count = 0;
  while (dU > 1e-6) {
    float U = 0;
    for (i = 0; i < NCELLS; i++) {
      Fx[i] = 0.0;
      Fy[i] = 0.0;
    }

    for (i = 0; i < NCELLS; i++) {
      xi = X[i];
      yi = Y[i];
      ri = Cells[i].r0 * 2;
      for (j = 0; j < NCELLS; j++) {
        if (j != i) {
          xj = X[j];
          yj = Y[j];
          rj = Cells[j].r0 * 2;
          dx = xj - xi;
          dx -= L * round(dx / L);
          dy = yj - yi;
          dy -= L * round(dy / L);
          dist = sqrt(dx * dx + dy * dy);
          if (dist < 0.0)
            dist *= -1;
          if (dist <= (ri + rj)) {
            ux = dx / dist;
            uy = dy / dist;
            ftmp = (1.0 - dist / (ri + rj)) / (ri + rj);
            fx = ftmp * ux;
            fy = ftmp * uy;
            Fx[i] -= fx;
            Fy[i] -= fy;
            Fy[j] += fy;
            Fx[j] += fx;
            U += 0.5 * (1 - (dist / (ri + rj)) * (1 - dist / (ri + rj)));
          }
        }
      }
    }
    for (int i = 0; i < NCELLS; i++) {
      X[i] += 0.01 * Fx[i];
      Y[i] += 0.01 * Fy[i];
    }
    dU = U - oldU;
    if (dU < 0.0)
      dU *= -1;
    oldU = U;
    count++;
    if (count > 1e5) {
      std::cerr << "Warning: Max timesteps for dispersion reached" << std::endl;
      break;
    }
  }
  for (i = 0; i < (int)NCELLS; i++) {
    std::array<float, 3> com = Cells[i].GetCOM();
    for (j = 0; j < (int)Cells[i].NV; j++) {
      Cells[i].Verts[j][0] -= com[0];
      Cells[i].Verts[j][1] -= com[1];
      Cells[i].Verts[j][0] -= X[i];
      Cells[i].Verts[j][1] -= Y[i];
    }
  }
}

void Tissue3D::CLEulerUpdate(int nsteps, float dt) {
  const int NF = Cell3D::NF;
  const int NV = Cell3D::NV;

  std::vector<cl_uint4> allFaces(NF);
  std::vector<cl_float4> allVerts(NCELLS * NV);
  std::vector<cl_float4> allForces(NCELLS * NV);
  std::vector<float> Kv;
  std::vector<float> Ka;
  std::vector<float> Ks;
  std::vector<float> v0;
  std::vector<float> a0;
  std::vector<float> l0;

  auto &c = Cells[0];
  for (int fi = 0; fi < NF; fi++) {
    allFaces[fi] = {{c.Faces[fi][0], c.Faces[fi][1], c.Faces[fi][2], 0}};
  }

  for (int ci = 0; ci < NCELLS; ci++) {
    Kv.push_back(Cells[ci].Kv);
    Ka.push_back(Cells[ci].Ka);
    Ks.push_back(Cells[ci].Ks);
    v0.push_back(Cells[ci].v0);
    a0.push_back(Cells[ci].a0);
    l0.push_back(sqrt((Cells[ci].a0 * 4.0f) / sqrt(3.0f)));
    for (int vi = 0; vi < NV; vi++) {
      int idx = ci * NV + vi;
      allVerts[idx] = {{Cells[ci].Verts[vi][0], Cells[ci].Verts[vi][1],
                        Cells[ci].Verts[vi][2], 0.0}};
      allForces[idx] = {{Cells[ci].Forces[vi][0], Cells[ci].Forces[vi][1],
                         Cells[ci].Forces[vi][2], 0.0}};
    }
  }

  // OpenCL Setup

  cl_int err = program.build({device}, "-cl-opt-disable -Werror");
  if (err != CL_SUCCESS) {
    std::cerr << "ERR: " << CL_SUCCESS << " : "
              << "kernel compilation failed:\n";
    std::string version = device.getInfo<CL_DEVICE_VERSION>();
    std::cerr << "OpenCL version:" << version << "\n";
    std::cerr << "platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cerr << "device:  " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cerr << "driver version: " << device.getInfo<CL_DRIVER_VERSION>()
              << "\n";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    exit(0);
  }

  cl::Buffer gpuFaces(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(cl_uint4) * NF, Cells[0].Faces.data());
  cl::Buffer gpuVerts(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(cl_float4) * NCELLS * NV, allVerts.data());
  cl::Buffer gpuForces(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(cl_float4) * NCELLS * NV, allForces.data());
  cl::Buffer gpuKv(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, Kv.data());
  cl::Buffer gpuKa(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, Ka.data());
  cl::Buffer gpuKs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, Ks.data());
  cl::Buffer gpuv0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, v0.data());
  cl::Buffer gpua0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, a0.data());
  cl::Buffer gpul0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, l0.data());

  // Create Kernels and set arguments (buffers)
  cl::Kernel VolumeUpdateKernel(program, "VolumeForceUpdate");
  VolumeUpdateKernel.setArg(0, gpuFaces);
  VolumeUpdateKernel.setArg(1, gpuVerts);
  VolumeUpdateKernel.setArg(2, gpuForces);
  VolumeUpdateKernel.setArg(3, NCELLS);
  VolumeUpdateKernel.setArg(4, gpuv0);
  VolumeUpdateKernel.setArg(5, gpuKv);

  cl::Kernel SurfaceAreaUpdateKernel(program, "SurfaceAreaForceUpdate");
  SurfaceAreaUpdateKernel.setArg(0, gpuFaces);
  SurfaceAreaUpdateKernel.setArg(1, gpuVerts);
  SurfaceAreaUpdateKernel.setArg(2, gpuForces);
  SurfaceAreaUpdateKernel.setArg(3, NCELLS);
  SurfaceAreaUpdateKernel.setArg(4, gpuKa);
  SurfaceAreaUpdateKernel.setArg(5, gpul0);

  cl::Kernel StickToSurfaceUpdate(program, "StickToSurface");
  StickToSurfaceUpdate.setArg(0, gpuFaces);
  StickToSurfaceUpdate.setArg(1, gpuVerts);
  StickToSurfaceUpdate.setArg(2, gpuForces);
  StickToSurfaceUpdate.setArg(3, NCELLS);
  StickToSurfaceUpdate.setArg(4, gpuKs);
  StickToSurfaceUpdate.setArg(5, gpul0);

  cl::Kernel RepellingForces(program, "RepellingForces");
  RepellingForces.setArg(0, gpuFaces);
  RepellingForces.setArg(1, gpuVerts);
  RepellingForces.setArg(2, gpuForces);
  RepellingForces.setArg(3, NCELLS);
  RepellingForces.setArg(4, gpul0);
  RepellingForces.setArg(5, Kre);
  RepellingForces.setArg(6, PBC);
  RepellingForces.setArg(7, L);
  /*cl::Kernel AllVertAttraction(program,"AllVertAttraction");
  AllVertAttraction.setArg(0, gpuVerts);
  AllVertAttraction.setArg(1, gpuForces);
  AllVertAttraction.setArg(2, gpul0);
  AllVertAttraction.setArg(3, L);
  AllVertAttraction.setArg(4, NCELLS);
  AllVertAttraction.setArg(5, PBC);
  AllVertAttraction.setArg(6, Kat);
  */

  cl::Kernel EulerUpdate(program, "EulerPosition");
  EulerUpdate.setArg(0, gpuVerts);
  EulerUpdate.setArg(1, gpuForces);
  EulerUpdate.setArg(2, dt);

  cl::NDRange globalSize(NF, NCELLS);
  cl::CommandQueue queue(context, device);

  // Run the kernels
  for (int step = 0; step < nsteps; step++) {
    queue.enqueueNDRangeKernel(VolumeUpdateKernel, cl::NullRange, globalSize);
    queue.enqueueNDRangeKernel(SurfaceAreaUpdateKernel, cl::NullRange,
                               globalSize);
    queue.enqueueNDRangeKernel(StickToSurfaceUpdate, cl::NullRange, globalSize);
    queue.enqueueNDRangeKernel(RepellingForces, cl::NullRange, globalSize);
    // queue.enqueueNDRangeKernel(AllVertAttraction,cl::NullRange, globalSize);
    if (step == nsteps - 1) {
      queue.enqueueReadBuffer(gpuForces, CL_TRUE, 0,
                              sizeof(std::array<float, 3>) * NV * NCELLS,
                              allForces.data());
    }
    queue.enqueueNDRangeKernel(EulerUpdate, cl::NullRange,
                               cl::NDRange(NV, NCELLS));
  }
  // Read the results
  queue.enqueueReadBuffer(gpuVerts, CL_TRUE, 0,
                          sizeof(std::array<float, 3>) * NV * NCELLS,
                          allVerts.data());

  // Update the cells
  for (int ci = 0; ci < NCELLS; ci++) {
    for (int vi = 0; vi < NV; vi++) {
      int idx = ci * NV + vi;
      Cells[ci].Verts[vi] = {allVerts[idx].s[0], allVerts[idx].s[1],
                             allVerts[idx].s[2]};
      Cells[ci].Forces[vi] = {allForces[idx].s[0], allForces[idx].s[1],
                              allForces[idx].s[2]};
    }
  }
  for (int ci = 0; ci < NCELLS; ci++) {
    Cells[ci].Volume = Cells[ci].GetVolume();
    Cells[ci].SurfaceArea = Cells[ci].GetSurfaceArea();
  }
}
} // namespace DPM
