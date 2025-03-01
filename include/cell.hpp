#include<vector>
#include<array>

#ifndef __CELL__
#define __CELL__
namespace DPM{
  struct Cell2D{
    int NV;
    float calA0;
    float l0;
    float r0;
    float v0;
    float Ka;
    float Kb;
    float Kl;
    float Area;
    float COMX;
    float COMY;
    std::vector<int> ip1;
    std::vector<int> im1;
    float vmin;
    float Dr;
    float Ds;
    float a0;
    float psi;
    float U;
    float Ks;
    std::vector<float> l1;
    std::vector<float> l2;
    std::vector<float> radii;
    std::vector<int> NearestVertexIdx;
    std::vector<int> NearestCellIdx;
    std::vector<std::array<float,2>> Verticies;
    Cell2D(float x0, float y0, float calA, int NV, float r0);
    Cell2D(std::vector<std::array<float,2>> Verticies);

    void SetCellVelocity(float v);
    void UpdateDirectorDiffusion(float dt);
    float GetArea();
  };
  class Cell3D{
    public:
    static const int NV = 162;
    static const int NF = 320;
    int calA0;
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
    std::vector<std::array<int,4>> Faces;
    void CLShapeEuler(int nsteps, float dt);
    float GetVolume();
    float GetSurfaceArea();
    std::array<std::array<float,NV>,3> GetPositions();
    std::array<std::array<float,NV>,3> GetForces();
    Cell3D(std::array<float,3> starting_point, float CalA0, float r0);
    
    private:
    std::vector<std::vector<int>> midpointCache;
    int AddMiddlePoint(int p1, int p2);
  };
}

#endif
