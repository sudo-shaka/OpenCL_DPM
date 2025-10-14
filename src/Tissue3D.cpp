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

  // Input validation
  if (nsteps <= 0) {
    std::cerr << "[ERROR] Invalid nsteps: " << nsteps << std::endl;
    throw std::invalid_argument("nsteps must be positive");
  }
  if (dt <= 0.0f || dt > 0.1f) {
    std::cerr << "[ERROR] Invalid dt: " << dt
              << " (should be in range (0, 0.1])" << std::endl;
    throw std::invalid_argument("dt must be positive and reasonable");
  }
  if (NCELLS <= 0) {
    std::cerr << "[ERROR] Invalid NCELLS: " << NCELLS << std::endl;
    throw std::invalid_argument("NCELLS must be positive");
  }

  // Get device properties for validation
  cl_int err = CL_SUCCESS;
  std::vector<cl_uint3> allFaces(NF);
  std::vector<cl_float3> allVerts(NCELLS * NV);
  std::vector<cl_float3> allForces(NCELLS * NV);
  std::vector<float> Kv, Ka, Ks, v0, a0, l0, volumes;

  auto &c = Cells[0];
  for (int fi = 0; fi < NF; fi++) {
    allFaces[fi] = {{c.Faces[fi][0], c.Faces[fi][1], c.Faces[fi][2], 0}};

    // Validate face indices
    if (c.Faces[fi][0] >= NV || c.Faces[fi][1] >= NV || c.Faces[fi][2] >= NV) {
      std::cerr << "[ERROR] Invalid face index at face " << fi << ": ("
                << c.Faces[fi][0] << "," << c.Faces[fi][1] << ","
                << c.Faces[fi][2] << ")" << std::endl;
      throw std::runtime_error("Invalid face indices");
    }
  }

  for (int ci = 0; ci < NCELLS; ci++) {
    // Validate cell parameters
    if (Cells[ci].Kv <= 0 || Cells[ci].Ka <= 0 || Cells[ci].Ks <= 0) {
      std::cerr << "[ERROR] Invalid spring constants for cell " << ci
                << ": Kv=" << Cells[ci].Kv << ", Ka=" << Cells[ci].Ka
                << ", Ks=" << Cells[ci].Ks << std::endl;
      throw std::runtime_error("Invalid spring constants");
    }
    if (Cells[ci].v0 <= 0 || Cells[ci].a0 <= 0) {
      std::cerr << "[ERROR] Invalid reference values for cell " << ci
                << ": v0=" << Cells[ci].v0 << ", a0=" << Cells[ci].a0
                << std::endl;
      throw std::runtime_error("Invalid reference values");
    }

    Kv.push_back(Cells[ci].Kv);
    Ka.push_back(Cells[ci].Ka);
    Ks.push_back(Cells[ci].Ks);
    v0.push_back(Cells[ci].v0);
    a0.push_back(Cells[ci].a0);
    l0.push_back(sqrt(4.0f * Cells[ci].a0) / sqrt(3.0f));
    volumes.push_back(0.0f);

    for (int vi = 0; vi < NV; vi++) {
      int idx = ci * NV + vi;

      // Validate vertex coordinates
      for (int coord = 0; coord < 3; coord++) {
        if (!std::isfinite(Cells[ci].Verts[vi][coord])) {
          std::cerr << "[ERROR] Non-finite vertex coordinate at cell " << ci
                    << ", vertex " << vi << ", coord " << coord << ": "
                    << Cells[ci].Verts[vi][coord] << std::endl;
          throw std::runtime_error("Non-finite vertex coordinates");
        }
      }

      allVerts[idx] = {{Cells[ci].Verts[vi][0], Cells[ci].Verts[vi][1],
                        Cells[ci].Verts[vi][2]}};
      allForces[idx] = {{0.0f, 0.0f, 0.0f}};
    }
  }

  err = program.build({device}, "-cl-opt-disable -Werror");
  if (err != CL_SUCCESS) {
    std::cerr << "[ERROR] Kernel compilation failed with error code: " << err
              << std::endl;
    std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::cerr << "[ERROR] Build log:\n" << buildLog << std::endl;
    throw std::runtime_error("Kernel compilation failed");
  }
  try {
    cl::Buffer gpuFaces(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_uint3) * NF, allFaces.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create faces buffer: " << err
                << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpuVerts(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_float3) * NCELLS * NV, allVerts.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create vertices buffer: " << err
                << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpuForces(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         sizeof(cl_float3) * NCELLS * NV, allForces.data(),
                         &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create forces buffer: " << err
                << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpuKv(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * NCELLS, Kv.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create Kv buffer: " << err << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpuKa(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * NCELLS, Ka.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create Ka buffer: " << err << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpuKs(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * NCELLS, Ks.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create Ks buffer: " << err << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpuv0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * NCELLS, v0.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create v0 buffer: " << err << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpua0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * NCELLS, a0.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create a0 buffer: " << err << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpul0(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * NCELLS, l0.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create l0 buffer: " << err << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    cl::Buffer gpuVolumes(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * NCELLS, volumes.data(), &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create volumes buffer: " << err
                << std::endl;
      throw std::runtime_error("Buffer creation failed");
    }

    // Create kernels with error checking
    cl::Kernel VolumeUpdateKernel(program, "VolumeForceUpdate", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create VolumeForceUpdate kernel: " << err
                << std::endl;
      throw std::runtime_error("Kernel creation failed");
    }
    VolumeUpdateKernel.setArg(0, gpuFaces);
    VolumeUpdateKernel.setArg(1, gpuVerts);
    VolumeUpdateKernel.setArg(2, gpuForces);
    VolumeUpdateKernel.setArg(3, NCELLS);
    VolumeUpdateKernel.setArg(4, gpuKv);
    VolumeUpdateKernel.setArg(5, gpuv0);
    VolumeUpdateKernel.setArg(6, gpuVolumes);

    cl::Kernel SurfaceAreaUpdateKernel(program, "SurfaceAreaForceUpdate", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create SurfaceAreaForceUpdate kernel: "
                << err << std::endl;
      throw std::runtime_error("Kernel creation failed");
    }
    SurfaceAreaUpdateKernel.setArg(0, gpuFaces);
    SurfaceAreaUpdateKernel.setArg(1, gpuVerts);
    SurfaceAreaUpdateKernel.setArg(2, gpuForces);
    SurfaceAreaUpdateKernel.setArg(3, NCELLS);
    SurfaceAreaUpdateKernel.setArg(4, gpuKa);
    SurfaceAreaUpdateKernel.setArg(5, gpua0);
    SurfaceAreaUpdateKernel.setArg(6, gpul0);

    cl::Kernel StickToSurfaceUpdate(program, "StickToSurface", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create StickToSurface kernel: " << err
                << std::endl;
      throw std::runtime_error("Kernel creation failed");
    }
    StickToSurfaceUpdate.setArg(0, gpuFaces);
    StickToSurfaceUpdate.setArg(1, gpuVerts);
    StickToSurfaceUpdate.setArg(2, gpuForces);
    StickToSurfaceUpdate.setArg(3, NCELLS);
    StickToSurfaceUpdate.setArg(4, gpuKs);
    StickToSurfaceUpdate.setArg(5, gpul0);

    cl::Kernel RepellingForces(program, "RepellingForces", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create RepellingForces kernel: " << err
                << std::endl;
      throw std::runtime_error("Kernel creation failed");
    }
    RepellingForces.setArg(0, gpuFaces);
    RepellingForces.setArg(1, gpuVerts);
    RepellingForces.setArg(2, gpuForces);
    RepellingForces.setArg(3, NCELLS);
    RepellingForces.setArg(4, gpul0);
    RepellingForces.setArg(5, Kre);
    RepellingForces.setArg(6, PBC);
    RepellingForces.setArg(7, L);

    cl::Kernel ClearForces(program, "ClearForces", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create ClearForces kernel: " << err
                << std::endl;
      throw std::runtime_error("Kernel creation failed");
    }
    ClearForces.setArg(0, gpuForces);

    cl::Kernel EulerUpdate(program, "EulerPosition", &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create EulerPosition kernel: " << err
                << std::endl;
      throw std::runtime_error("Kernel creation failed");
    }
    EulerUpdate.setArg(0, gpuVerts);
    EulerUpdate.setArg(1, gpuForces);
    EulerUpdate.setArg(2, dt);
    EulerUpdate.setArg(3, NCELLS);

    cl::NDRange faceCellSize(NF, NCELLS);
    cl::NDRange vertCellSize(NV, NCELLS);
    cl::NDRange globalVertSize(NCELLS * NV);
    cl::CommandQueue queue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to create command queue: " << err
                << std::endl;
      throw std::runtime_error("Command queue creation failed");
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Run simulation
    for (int step = 0; step < nsteps; step++) {
      // Clear forces at start of each step
      err = queue.enqueueNDRangeKernel(ClearForces, cl::NullRange,
                                       globalVertSize);
      if (err != CL_SUCCESS) {
        std::cerr << "[ERROR] Failed to enqueue ClearForces kernel in step "
                  << step << ": " << err << std::endl;
        throw std::runtime_error("Kernel execution failed");
      }

      // Accumulate all forces
      err = queue.enqueueNDRangeKernel(VolumeUpdateKernel, cl::NullRange,
                                       faceCellSize);
      if (err != CL_SUCCESS) {
        std::cerr << "[ERROR] Failed to enqueue VolumeUpdateKernel in step "
                  << step << ": " << err << std::endl;
        throw std::runtime_error("Kernel execution failed");
      }

      err = queue.enqueueNDRangeKernel(SurfaceAreaUpdateKernel, cl::NullRange,
                                       faceCellSize);
      if (err != CL_SUCCESS) {
        std::cerr
            << "[ERROR] Failed to enqueue SurfaceAreaUpdateKernel in step "
            << step << ": " << err << std::endl;
        throw std::runtime_error("Kernel execution failed");
      }

      err = queue.enqueueNDRangeKernel(StickToSurfaceUpdate, cl::NullRange,
                                       faceCellSize);
      if (err != CL_SUCCESS) {
        std::cerr << "[ERROR] Failed to enqueue StickToSurfaceUpdate in step "
                  << step << ": " << err << std::endl;
        throw std::runtime_error("Kernel execution failed");
      }

      err = queue.enqueueNDRangeKernel(RepellingForces, cl::NullRange,
                                       globalVertSize);
      if (err != CL_SUCCESS) {
        std::cerr << "[ERROR] Failed to enqueue RepellingForces in step "
                  << step << ": " << err << std::endl;
        throw std::runtime_error("Kernel execution failed");
      }

      // Update positions
      err =
          queue.enqueueNDRangeKernel(EulerUpdate, cl::NullRange, vertCellSize);
      if (err != CL_SUCCESS) {
        std::cerr << "[ERROR] Failed to enqueue EulerUpdate in step " << step
                  << ": " << err << std::endl;
        throw std::runtime_error("Kernel execution failed");
      }

      if (step == nsteps - 1) {
        err = queue.enqueueReadBuffer(gpuForces, CL_TRUE, 0,
                                      sizeof(cl_float3) * NV * NCELLS,
                                      allForces.data());
        if (err != CL_SUCCESS) {
          std::cerr << "[ERROR] Failed to read forces buffer: " << err
                    << std::endl;
          throw std::runtime_error("Buffer read failed");
        }
      }

      // Synchronize every 1000 steps to catch errors early
      if (step % 1000 == 0) {
        err = queue.finish();
        if (err != CL_SUCCESS) {
          std::cerr << "[ERROR] Failed to finish queue in step " << step << ": "
                    << err << std::endl;
          throw std::runtime_error("Queue finish failed");
        }
      }
    }

    err = queue.finish(); // Ensure all operations complete
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to finish final queue operations: " << err
                << std::endl;
      throw std::runtime_error("Queue finish failed");
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);
    std::cout << nsteps << " timesteps completed in " << duration.count()
              << " ms" << std::endl;
    std::cout << "Average time per step: "
              << (duration.count() / static_cast<double>(nsteps)) << " ms"
              << std::endl;

    // Read final results
    err = queue.enqueueReadBuffer(
        gpuVerts, CL_TRUE, 0, sizeof(cl_float3) * NV * NCELLS, allVerts.data());
    if (err != CL_SUCCESS) {
      std::cerr << "[ERROR] Failed to read vertices buffer: " << err
                << std::endl;
      throw std::runtime_error("Buffer read failed");
    }

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
    throw;
  }

  for (int ci = 0; ci < NCELLS; ci++) {
    for (int vi = 0; vi < NV; vi++) {
      int idx = ci * NV + vi;

      // Validate results before copying back
      for (int coord = 0; coord < 3; coord++) {
        if (!std::isfinite(allVerts[idx].s[coord])) {
          std::cerr << "[ERROR] Non-finite result vertex at cell " << ci
                    << ", vertex " << vi << ", coord " << coord << ": "
                    << allVerts[idx].s[coord] << std::endl;
          throw std::runtime_error("Non-finite simulation results");
        }
        if (!std::isfinite(allForces[idx].s[coord])) {
          std::cerr << "[WARNING] Non-finite force at cell " << ci
                    << ", vertex " << vi << ", coord " << coord << ": "
                    << allForces[idx].s[coord] << std::endl;
        }
      }

      Cells[ci].Verts[vi] = {allVerts[idx].s[0], allVerts[idx].s[1],
                             allVerts[idx].s[2]};
      Cells[ci].Forces[vi] = {allForces[idx].s[0], allForces[idx].s[1],
                              allForces[idx].s[2]};
    }

    try {
      Cells[ci].Volume = Cells[ci].GetVolume();
      Cells[ci].SurfaceArea = Cells[ci].GetSurfaceArea();

      // Validate computed properties
      if (!std::isfinite(Cells[ci].Volume) || Cells[ci].Volume <= 0) {
        std::cerr << "[WARNING] Invalid volume for cell " << ci << ": "
                  << Cells[ci].Volume << std::endl;
      }
      if (!std::isfinite(Cells[ci].SurfaceArea) || Cells[ci].SurfaceArea <= 0) {
        std::cerr << "[WARNING] Invalid surface area for cell " << ci << ": "
                  << Cells[ci].SurfaceArea << std::endl;
      }

    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Error computing cell properties for cell " << ci
                << ": " << e.what() << std::endl;
      throw;
    }
  }
}
} // namespace DPM