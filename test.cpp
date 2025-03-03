#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <vector>
#include "cell.hpp"
#include "Tissue.hpp"

int main(){
  float r0 = 1.8f;
  std::array<float,3> startp1 = {7.0,6.0,1.3};
  std::array<float,3> startp2 = {4,6.0,1.3};
  DPM::Cell3D Cell(startp1, 1.05f, r0);
  DPM::Cell3D Cell2(startp2, 1.05f, r0);
  Cell.Kv = 5.0;
  Cell.Ka = 1.5;
  Cell.Ks = 0.0;
  Cell2.Kv = 5.0;
  Cell2.Ka = 1.5;
  Cell2.Ks = 0.0;

  Cell.Kv = 1;
  Cell.Ka = 1;
  Cell.Ks = 1;

  Cell2.Kv = 1;
  Cell2.Ka = 1;
  Cell2.Ks = 1;

  std::vector<DPM::Cell3D> Cells;
  for(int ci=0;ci<15;ci++){
    Cells.push_back(Cell);
    Cells.push_back(Cell2);
  }


  DPM::Tissue3D Tissue(Cells, 0.35);
  Tissue.Kre = 1.0f;
  Tissue.Disperse2D();
  Tissue.CLEulerUpdate(1000,0.005);

  for(int i=0;i<(int)Tissue.NCELLS;i++){
    for(int j=0;j<(int)Tissue.Cells[i].NV;j++){
      std::cout << Tissue.Cells[i].Verts[j][0] << "," << Tissue.Cells[i].Verts[j][1] << "," << Tissue.Cells[i].Verts[j][2] << std::endl;
    }
  }

  /*Cell.Kv = 1;
  Cell.Ka = 1;
  Cell.Ks = 1;

  //Cell.CLShapeEuler(1, 0.001);

  for(int i=0;i<(int)Cell.NV;i++){
    std::cout << Cell.Verts[i][0] << "," << Cell.Verts[i][1] << "," << Cell.Verts[i][2] << std::endl;
  }

  for(int step=0;step<20;step++){
    Cell.CLShapeEuler(50, 0.005);

    for(int i=0;i<(int)Cell.NV;i++){
      std::cout << Cell.Verts[i][0] << "," << Cell.Verts[i][1] << "," << Cell.Verts[i][2] << std::endl;
    }

  }

  for(int i= 0;i<(int)Cell.NV;i++){
    std::cout << Cell.Verts[i][0] << "," << Cell.Verts[i][1] << "," << Cell.Verts[i][2] << std::endl;
  }*/

  return 0;
}
