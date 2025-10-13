#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "Tissue.hpp"
#include "cell.hpp"
#include "readKernel.hpp"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace DPM {
Tissue2D::Tissue2D(std::vector<Cell2D> inputCells, float packingFraction) {
  cells = inputCells;
  NCELLS = cells.size();
  PBC = true;
  Kre = 1.0;
  Kat = 0.0;
  phi0 = packingFraction;

  float totalArea = 0.0f;
  maxNV = cells[0].NV;
  for (int ci = 0; ci < (int)cells.size(); ci++) {
    totalArea += cells[ci].GetArea();
    if (cells[ci].NV > (unsigned int)maxNV) {
      maxNV = cells[ci].NV;
    }
  }
  L = sqrt(totalArea) / phi0;
  Kre = 1.0f;
  Kat = 0.0f;

  platform = cl::Platform::getDefault();
  device = cl::Device::getDefault();
  context = cl::Context({device});
  std::string source = readKernelSource("./shaders/Cell2D_kernel.cl");
  program = cl::Program(context, source);
}

void Tissue2D::Disperse() {
  std::vector<float> X, Y, Fx, Fy;
  X.resize(NCELLS);
  Y.resize(NCELLS);
  Fx.resize(NCELLS);
  Fy.resize(NCELLS);
  float ri, xi, yi, xj, yj, dx, dy, rj, dist;
  float ux, uy, ftmp, fx, fy, U = 0.0f;
  int i, j, count = 0;
  for (i = 0; i < NCELLS; i++) {
    X[i] = drand48() * L;
    Y[i] = drand48() * L;
  }
  float oldU = 100, dU = 100;
  while (dU > 1e-6) {
    U = 0;
    for (i = 0; i < NCELLS; i++) {
      Fx[i] = 0.0;
      Fy[i] = 0.0;
    }
    for (i = 0; i < NCELLS; i++) {
      xi = X[i];
      yi = Y[i];
      ri = cells[i].r0;
      for (j = 0; j < NCELLS; j++) {
        if (j != i) {
          xj = X[j];
          yj = Y[j];
          rj = cells[j].r0;
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
      break;
    }
  }
  for (int i = 0; i < NCELLS; i++) {
    for (j = 0; j < (int)cells[i].NV; j++) {
      cells[i].Verticies[j][0] =
          cells[i].r0 * (cos(2.0 * M_PI * (j + 1) / cells[i].NV)) + X[i];
      cells[i].Verticies[j][1] =
          cells[i].r0 * (sin(2.0 * M_PI * (j + 1) / cells[i].NV)) + Y[i];
    }
  }
}

void Tissue2D::CLEulerUpdate(int nsteps, float dt) {
  std::vector<std::array<float, 2>> allVerts;
  allVerts.resize(maxNV * NCELLS);
  std::vector<std::array<float, 2>> allForces;
  allForces.resize(maxNV * NCELLS);
  std::vector<float> Ka;
  std::vector<float> l0;
  std::vector<float> a0;
  std::vector<float> r0;
  std::vector<float> Kl;
  std::vector<float> Kb;
  std::vector<unsigned int> NV;

  for (int ci = 0; ci < (int)NCELLS; ci++) {
    Ka.push_back(cells[ci].Ka);
    Kl.push_back(cells[ci].Kl);
    Kb.push_back(cells[ci].Kb);
    l0.push_back(cells[ci].l0);
    a0.push_back(cells[ci].a0);
    NV.push_back(cells[ci].NV);
    r0.push_back(cells[ci].r0);
    for (unsigned int vi = 0; vi < cells[ci].NV; vi++) {
      allVerts[ci * maxNV + vi] = cells[ci].Verticies[vi];
    }
  }

  cl_int err = program.build({device});
  if (err != CL_SUCCESS) {
    std::cerr << "Error compiling kernel!" << std::endl;
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    exit(0);
  }
  cl::Buffer gpuVerts(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(std::array<float, 2>) * NCELLS * maxNV,
                      allVerts.data());
  cl::Buffer gpuForces(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(std::array<float, 2>) * NCELLS * maxNV,
                       allForces.data());
  cl::Buffer gpuNV(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(unsigned int) * NCELLS, NV.data());
  cl::Buffer gpuKa(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, Ka.data());
  cl::Buffer gpuKl(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, Kl.data());
  cl::Buffer gpuKb(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, Kb.data());
  cl::Buffer gpua0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, a0.data());
  cl::Buffer gpul0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, l0.data());
  cl::Buffer gpur0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * NCELLS, r0.data());

  cl::Kernel AreaForceUpdates(program, "AreaForceUpdates");
  AreaForceUpdates.setArg(0, gpuVerts);
  AreaForceUpdates.setArg(1, gpuForces);
  AreaForceUpdates.setArg(2, gpuNV);
  AreaForceUpdates.setArg(3, gpua0);
  AreaForceUpdates.setArg(4, gpuKa);
  cl::Kernel PerimeterForceUpdates(program, "PerimeterForceUpdates");
  PerimeterForceUpdates.setArg(0, gpuVerts);
  PerimeterForceUpdates.setArg(1, gpuForces);
  PerimeterForceUpdates.setArg(2, gpuNV);
  PerimeterForceUpdates.setArg(3, gpuKl);
  PerimeterForceUpdates.setArg(4, gpua0);
  PerimeterForceUpdates.setArg(5, gpul0);
  cl::Kernel BendingForceUpdates(program, "BendingForceUpdates");
  BendingForceUpdates.setArg(0, gpuVerts);
  BendingForceUpdates.setArg(1, gpuForces);
  BendingForceUpdates.setArg(2, gpuNV);
  BendingForceUpdates.setArg(3, gpuKb);
  BendingForceUpdates.setArg(4, gpua0);
  BendingForceUpdates.setArg(5, gpul0);
  cl::Kernel RepulsiveForceUpdate(program, "RepulsionForceUpdate");
  RepulsiveForceUpdate.setArg(0, gpuVerts);
  RepulsiveForceUpdate.setArg(1, gpuForces);
  RepulsiveForceUpdate.setArg(2, gpuNV);
  RepulsiveForceUpdate.setArg(3, gpur0);
  RepulsiveForceUpdate.setArg(4, (int)PBC);
  RepulsiveForceUpdate.setArg(5, L);
  RepulsiveForceUpdate.setArg(6, Kre);
  cl::Kernel AttractionForceUpdate(program, "AttractionForceUpdate");
  AttractionForceUpdate.setArg(0, gpuVerts);
  AttractionForceUpdate.setArg(1, gpuForces);
  AttractionForceUpdate.setArg(2, gpuNV);
  AttractionForceUpdate.setArg(3, gpul0);
  AttractionForceUpdate.setArg(4, (int)PBC);
  AttractionForceUpdate.setArg(5, L);
  AttractionForceUpdate.setArg(6, Kat);
  cl::Kernel EulerUpdate(program, "EulerUpdate");
  EulerUpdate.setArg(0, gpuVerts);
  EulerUpdate.setArg(1, gpuForces);
  EulerUpdate.setArg(2, gpuNV);
  EulerUpdate.setArg(3, dt);

  cl::NDRange globalSize(NCELLS, maxNV);
  cl::CommandQueue queue(context, device);

  for (int step = 0; step < nsteps; step++) {
    queue.enqueueNDRangeKernel(AreaForceUpdates, cl::NullRange, globalSize);
    queue.enqueueNDRangeKernel(PerimeterForceUpdates, cl::NullRange,
                               globalSize);
    queue.enqueueNDRangeKernel(BendingForceUpdates, cl::NullRange, globalSize);
    queue.enqueueNDRangeKernel(AttractionForceUpdate, cl::NullRange,
                               globalSize);
    queue.enqueueNDRangeKernel(RepulsiveForceUpdate, cl::NullRange, globalSize);
    if (step == nsteps - 1) {
      queue.enqueueReadBuffer(gpuForces, CL_TRUE, 0,
                              sizeof(std::array<float, 2>) * NCELLS * maxNV,
                              allForces.data());
    }
    queue.enqueueNDRangeKernel(EulerUpdate, cl::NullRange, globalSize);
  }

  queue.enqueueReadBuffer(gpuVerts, CL_TRUE, 0,
                          sizeof(std::array<float, 2>) * NCELLS * maxNV,
                          allVerts.data());

  for (int ci = 0; ci < (int)NCELLS; ci++) {
    for (unsigned int vi = 0; vi < cells[ci].NV; vi++) {
      cells[ci].Verticies[vi] = allVerts[ci * maxNV + vi];
      cells[ci].Forces[vi] = allForces[ci * maxNV + vi];
    }
  }
}

} // namespace DPM
