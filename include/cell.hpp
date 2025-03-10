#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include<array>
#include<vector>
#include <CL/cl.h>
#include <CL/opencl.hpp>

#ifndef __CELL__
#define __CELL__
namespace DPM{
  struct Cell2D{
    unsigned int NV;
    float calA0;
    float a0;
    float l0;
    float r0;
    float Ka;
    float Kb;
    float Kl;
    float Ks;
    std::vector<std::array<float,2>> Verticies;
    std::vector<std::array<float,2>> Forces;
    Cell2D(float x0, float y0, float calA,unsigned int NV, float a0);
    Cell2D(std::vector<std::array<float,2>> Verticies);
    float GetArea();
    float GetPerim();
  };
  class Cell3D{
    public:
    static const unsigned int NV = 162;
    static const unsigned int NF = 320;
    float calA0;
    float r0;
    float v0;
    float sa0;
    float a0;
    float Kv;
    float Ka;
    float Ks;
    float Volume;
    float SurfaceArea;
    std::vector<std::array<float,4>> Verts;
    std::vector<std::array<float,4>> Forces;
    std::vector<std::array<unsigned int,4>> Faces;

    void CLShapeEuler(unsigned int nsteps, float dt);
    float GetVolume();
    float GetSurfaceArea();
    std::array<std::array<float,NV>,3> GetPositions();
    std::array<std::array<float,162>,3> GetVesselPositions(float L);
    std::array<std::array<float,NV>,3> GetForces();
    std::array<std::array<int,3>,NF> GetFaces();
    std::array<float,3>  GetCOM();
    Cell3D(std::array<float,3> starting_point, float CalA0, float r0);

    private:
    std::string kernelSource;
    cl::Platform platform;
    cl::Device device;
    cl::Program program;
    cl::Context context;
    std::vector<std::vector<int>> midpointCache;
    unsigned int AddMiddlePoint(unsigned int p1, unsigned int p2);
  };
}

#endif
