#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_TARGET_OPENCL_VERSION 300
#include "cell.hpp"
#include "readKernel.hpp"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <array>
#include <cmath>
#include <vector>

namespace DPM {
Cell2D::Cell2D(float x0, float y0, float CalA, unsigned int numVerts,
               float _r0) {
  // Initialize number of vertices
  NV = numVerts;
  r0 = _r0;
  // Calculate initial shape parameters
  calA0 = CalA * (NV * tan(M_PI / NV) / M_PI);

  // Resize vectors to hold vertices and forces
  Verticies.resize(NV);
  Forces.resize(NV);

  // Initialize vertices positions and forces
  for (unsigned int i = 0; i < NV; i++) {
    Verticies[i][0] = r0 * (cos(2.0 * M_PI * (i + 1.0) / (float)NV)) + x0;
    Verticies[i][1] = r0 * (sin(2.0 * M_PI * (i + 1.0) / (float)NV)) + y0;
    Forces[i][0] = 0.0f;
    Forces[i][1] = 0.0f;
  }
  a0 = GetArea();
  l0 = 2.0 * sqrt(M_PI * calA0 * a0) / (float)NV;
}

float Cell2D::GetArea() {
  float Area = 0.0f;
  unsigned int j = NV - 1;
  for (unsigned int i = 0; i < NV; i++) {
    Area += 0.5 * ((Verticies[j][0] + Verticies[i][0]) *
                   (Verticies[j][1] - Verticies[i][1]));
    j = i;
  }
  if (Area < 0.0f) {
    Area = -Area;
  }
  return Area;
}

float Cell2D::GetPerim() {
  float dx, dy, dist = 0.0;
  for (unsigned int i = 0; i < NV - 1; i++) {
    dx = Verticies[i + 1][0] - Verticies[i][0];
    dy = Verticies[i + 1][1] - Verticies[i][1];
    dist += sqrt(dx * dx + dy * dy);
  }
  dx = Verticies[NV][0] - Verticies[0][0];
  dy = Verticies[NV][1] - Verticies[0][1];
  dist += sqrt(dx * dx + dy * dy);
  return dist;
}

Cell3D::Cell3D(std::array<float, 3> starting_point, float calA, float _r0) {

  calA0 = calA;
  r0 = _r0;
  Kv = 0.0f;
  Ka = 0.0f;
  Verts.resize(12);
  float t =
      (1 + sqrt(5)) / 2; // only need first 3, 0 at end is padded for GPU float4
  Verts[0] = {-1, t, 0};
  Verts[0] = {-1, t, 0};
  Verts[1] = {1, t, 0};
  Verts[2] = {-1, -t, 0};
  Verts[3] = {1, -t, 0};

  Verts[4] = {0, -1, t};
  Verts[5] = {0, 1, t};
  Verts[6] = {0, -1, -t};
  Verts[7] = {0, 1, -t};

  Verts[8] = {t, 0, -1};
  Verts[9] = {t, 0, 1};
  Verts[10] = {-t, 0, -1};
  Verts[11] = {-t, 0, 1};
  for (int i = 0; i < 12; i++) {
    float norm =
        std::sqrt(Verts[i][0] * Verts[i][0] + Verts[i][1] * Verts[i][1] +
                  Verts[i][2] * Verts[i][2]);
    Verts[i][0] /= norm;
    Verts[i][1] /= norm;
    Verts[i][2] /= norm;
  }
  Faces.push_back(std::array<unsigned int, 3>{0, 11, 5});
  Faces.push_back(std::array<unsigned int, 3>{0, 5, 1});
  Faces.push_back(std::array<unsigned int, 3>{0, 1, 7});
  Faces.push_back(std::array<unsigned int, 3>{0, 7, 10});
  Faces.push_back(std::array<unsigned int, 3>{0, 10, 11});

  // 5 adjacent faces
  Faces.push_back(std::array<unsigned int, 3>{1, 5, 9});
  Faces.push_back(std::array<unsigned int, 3>{5, 11, 4});
  Faces.push_back(std::array<unsigned int, 3>{11, 10, 2});
  Faces.push_back(std::array<unsigned int, 3>{10, 7, 6});
  Faces.push_back(std::array<unsigned int, 3>{7, 1, 8});

  // 5 faces around point 3
  Faces.push_back(std::array<unsigned int, 3>{3, 9, 4});
  Faces.push_back(std::array<unsigned int, 3>{3, 4, 2});
  Faces.push_back(std::array<unsigned int, 3>{3, 2, 6});
  Faces.push_back(std::array<unsigned int, 3>{3, 6, 8});
  Faces.push_back(std::array<unsigned int, 3>{3, 8, 9});

  // 5 adjacent faces
  Faces.push_back(std::array<unsigned int, 3>{4, 9, 5});
  Faces.push_back(std::array<unsigned int, 3>{2, 4, 11});
  Faces.push_back(std::array<unsigned int, 3>{6, 2, 10});
  Faces.push_back(std::array<unsigned int, 3>{8, 6, 7});
  Faces.push_back(std::array<unsigned int, 3>{9, 8, 1});

  std::array<unsigned int, 3> newF;
  std::vector<std::array<unsigned int, 3>> newFaces;
  unsigned int f = 2;
  for (unsigned int i = 0; i < f; i++) {
    unsigned int steps = Faces.size();
    for (unsigned int j = 0; j < steps; j++) {
      unsigned int a = Cell3D::AddMiddlePoint(Faces[j][0], Faces[j][1]);
      unsigned int b = Cell3D::AddMiddlePoint(Faces[j][1], Faces[j][2]);
      unsigned int c = Cell3D::AddMiddlePoint(Faces[j][2], Faces[j][0]);
      newF = {Faces[j][0], a, c};
      newFaces.push_back(newF);
      newF = {Faces[j][1], b, a};
      newFaces.push_back(newF);
      newF = {Faces[j][2], c, b};
      newFaces.push_back(newF);
      newF = {a, b, c};
      newFaces.push_back(newF);
    }
    Faces = newFaces;
    newFaces.clear();
    newFaces.shrink_to_fit();
  }
  Forces.resize(NV);
  for (unsigned int vi = 0; vi < NV; vi++) {
    Verts[vi][0] *= r0;
    Verts[vi][1] *= r0;
    Verts[vi][2] *= r0;
    Verts[vi][0] += starting_point[0];
    Verts[vi][1] += starting_point[1];
    Verts[vi][2] += starting_point[2];
    Forces[vi] = {0, 0, 0};
  }
  v0 = (4.0f / 3.0f) * M_PI * pow(r0, 3);
  sa0 = pow((6 * sqrt(M_PI) * v0 * calA), (2.0f / 3.0f));
  a0 = (sa0 / (float)NF);
  Volume = GetVolume();
  SurfaceArea = GetSurfaceArea();
}

unsigned int Cell3D::AddMiddlePoint(unsigned int p1, unsigned int p2) {
  int key;
  int i;
  if (p1 < p2) {
    key = floor((p1 + p2) * (p1 + p2 + 1) / 2) + p1;
  } else {
    key = floor((p1 + p2) * (p1 + p2 + 1) / 2) + p2;
  }
  for (i = 0; i < (int)midpointCache.size(); i++) {
    if (key == (int)midpointCache[i][0])
      return midpointCache[i][1];
  }

  std::array<float, 3> vert1 = Verts[p2], vert2 = Verts[p1];
  std::array<float, 3> middlePoint;
  for (int i = 0; i < 3; i++) {
    middlePoint[i] = vert1[i] + vert2[i];
    middlePoint[i] *= 0.5;
  }
  for (int i = 0; i < 3; i++) {
    float norm = std::sqrt(middlePoint[0] * middlePoint[0] +
                           middlePoint[1] * middlePoint[1] +
                           middlePoint[2] * middlePoint[2]);
    middlePoint[i] /= norm;
  }
  Verts.push_back(middlePoint);
  i = Verts.size() - 1;
  std::vector<int> cache;
  cache.resize(2);
  cache.shrink_to_fit();
  cache[0] = key;
  cache[1] = i;
  midpointCache.push_back(cache);
  return i;
}

float Cell3D::GetVolume() {
  float volume = 0.0;
  for (const auto &tri : Faces) {
    std::array<float, 3> v0 = Verts[tri[0]];
    std::array<float, 3> v1 = Verts[tri[1]];
    std::array<float, 3> v2 = Verts[tri[2]];
    std::array<float, 3> cross;
    cross[0] = v1[1] * v2[2] - v1[2] * v2[1];
    cross[1] = v1[2] * v2[0] - v1[0] * v2[2];
    cross[2] = v1[0] * v2[1] - v1[1] * v2[0];
    float partialVolume =
        cross[0] * v0[0] + cross[1] * v0[1] + cross[2] * v0[2];
    volume += partialVolume;
  }
  return std::abs(volume) / 6.0;
}

float Cell3D::GetSurfaceArea() {
  float SurfaceArea = 0.0;
  for (const auto &tri : Faces) {
    std::array<float, 3> v0 = Verts[tri[0]];
    std::array<float, 3> v1 = Verts[tri[1]];
    std::array<float, 3> v2 = Verts[tri[2]];
    std::array<float, 3> A = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    std::array<float, 3> B = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    std::array<float, 3> cross;
    cross[0] = A[1] * B[2] - A[2] * B[1];
    cross[1] = A[2] * B[0] - A[0] * B[2];
    cross[2] = A[0] * B[1] - A[1] * B[0];
    float partialArea = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                                  cross[2] * cross[2]);
    SurfaceArea += partialArea;
  }
  return SurfaceArea;
}

std::array<std::array<float, 162>, 3> Cell3D::GetPositions() {
  std::array<std::array<float, NV>, 3> positions;
  for (unsigned int i = 0; i < NV; i++) {
    positions[0][i] = Verts[i][0];
    positions[1][i] = Verts[i][1];
    positions[2][i] = Verts[i][2];
  }
  return positions;
}

std::array<std::array<float, 162>, 3> Cell3D::GetVesselPositions(float L) {
  std::array<std::array<float, NV>, 3> positions;
  float scale = (2.0 * M_PI) / L;
  float radius = L / (2 * M_PI);
  for (unsigned int vi = 0; vi < NV; vi++) {
    float theta = Verts[vi][0] * scale;
    positions[0][vi] = (radius - Verts[vi][2]) * cos(theta);
    positions[2][vi] = (radius - Verts[vi][2]) * sin(theta);
    positions[1][vi] = Verts[vi][1];
  }
  return positions;
}

std::array<std::array<float, 162>, 3> Cell3D::GetForces() {
  std::array<std::array<float, NV>, 3> forces;
  for (unsigned int i = 0; i < NV; i++) {
    forces[0][i] = Forces[i][0];
    forces[1][i] = Forces[i][1];
    forces[2][i] = Forces[i][2];
  }
  return forces;
}

std::array<std::array<int, 3>, 320> Cell3D::GetFaces() {
  std::array<std::array<int, 3>, NF> faces;
  for (unsigned int i = 0; i < NF; i++) {
    for (int j = 0; j < 3; j++) {
      faces[i][j] = (int)Faces[i][j];
    }
  }

  return faces;
}

std::array<float, 3> Cell3D::GetCOM() {
  std::array<float, 3> com = {0.0, 0.0, 0.0};
  for (unsigned int vi = 0; vi < NV; vi++) {
    com[0] += Verts[vi][0];
    com[1] += Verts[vi][1];
    com[2] += Verts[vi][2];
  }
  com[0] /= (float)NV;
  com[1] /= (float)NV;
  com[2] /= (float)NV;

  return com;
}

} // namespace DPM
