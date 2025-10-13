#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "cell.hpp"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <string>
#include <vector>

#ifndef __TISSUE__
#define __TISSUE__

namespace DPM {
class Tissue3D {
public:
  int NCELLS;
  int PBC; //(boolean) periodic boundary conditions
  float L;
  float Kre;
  float Kat;
  std::string attractionMethod;
  std::vector<DPM::Cell3D> Cells;
  cl::Platform platform;
  cl::Device device;
  cl::Program program;
  cl::Context context;

  Tissue3D(std::vector<DPM::Cell3D> cells, float phi0);

  void CLEulerUpdate(int nsteps, float dt);
  void Disperse2D();
  void Disperse3D();
  std::vector<std::vector<float>> GetVesselPositions(int ci);
};

class Tissue2D {
public:
  std::vector<Cell2D> cells;
  int NCELLS;
  bool PBC;
  float Kre;
  float Kat;
  float phi0;
  float L;
  Tissue2D(std::vector<DPM::Cell2D> cells, float phi0);
  void Disperse();
  void CLEulerUpdate(int nsteps, float dt);

private:
  int maxNV;
  std::string kernelSource;
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::Program program;
};
} // namespace DPM

#endif
