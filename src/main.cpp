#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <vector>
#include "cell.hpp"
#include "Tissue.hpp"

int main(){
  float r0 = 1.0f;
  std::array<float,3> startp = {0,0,r0};
  DPM::Cell3D Cell(startp, 1.2 ,r0);
  Cell.Kv = 0.5;
  Cell.Ka = 2.0;
  Cell.Ks = 0.0;

  std::vector<DPM::Cell3D> Cells;
  for(int i=0;i<5;i++){
    Cells.push_back(Cell);
  }
  DPM::Tissue3D Tissue(Cells, 0.1);

  Tissue.Disperse2D();
  Tissue.CLEulerUpdate(100,0.001);

  /*for(int i=0;i<(int)Tissue.NCELLS;i++){
    for(int j=0;j<(int)Tissue.Cells[i].NV;j++){
      std::cout << Tissue.Cells[i].Verts[j][0] << "," << Tissue.Cells[i].Verts[j][1] << "," << Tissue.Cells[i].Verts[j][2] << std::endl;
    }
  }*/

  Cell.CLShapeEuler(1000, 0.001);
  for(int i=0;i<(int)Cell.NV;i++){
    std::cout << Cell.Verts[i][0] << "," << Cell.Verts[i][1] << "," << Cell.Verts[i][2] << std::endl;
  }

  for(int step=0;step<50;step++){
    Cell.CLShapeEuler(1, 0.005);
  
    for(int i=0;i<(int)Cell.NV;i++){
      std::cout << Cell.Verts[i][0] << "," << Cell.Verts[i][1] << "," << Cell.Verts[i][2] << std::endl;
    }
  
  }

  for(int i= 0;i<(int)Cell.NV;i++){
    std::cout << Cell.Verts[i][0] << "," << Cell.Verts[i][1] << "," << Cell.Verts[i][2] << std::endl;
  }

  return 0;
}
