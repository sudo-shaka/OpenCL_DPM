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
  struct Cell3D{
    int NV;
    int calA0;
    float r0;
    float v0;
    float a0;
    float s0;
    float Kv;
    float Ka;
    float Kb;
    float Ks;
    int NF; //number of  faces
    float Volume;
    float SurfaceArea;
    std::vector<std::array<float,3>> Verts;
    std::vector<std::array<float,3>> Forces;
    std::vector<std::array<int,3>> Faces;
    std::vector<std::vector<int>> midpointCache;
    float l0;
    Cell3D(std::array<float,3> starting_point, float CalA0, float r0);
    int AddMiddlePoint(int p1, int p2);
  };
}

#endif
