#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "Tissue.hpp"
#include "cell.hpp"
#include <iostream>
#include <vector>

int main() {
  DPM::Cell2D Cell(0, 0, 1.05f, 32, 1.0);
  DPM::Cell2D Cell2(0, 0, 1.05f, 40, 1.5);
  Cell.Ka = 1.0;
  Cell.Kl = 1.0;
  Cell.Kb = 0.1;

  Cell2.Ka = 1.0;
  Cell2.Kl = 1.0;
  Cell2.Kb = 0.1;

  std::vector<DPM::Cell2D> Cells;
  /*for(int ci=0;ci<15;ci++){
    Cells.push_back(Cell);
    Cells.push_back(Cell2);
  }*/
  Cells.push_back(Cell);

  std::cout << Cell.GetArea() << std::endl;

  DPM::Tissue2D Tissue(Cells, 0.85);
  Tissue.Kre = 50.0f;
  Tissue.Disperse();

  /*  for(int i=0;i<(int)Tissue.NCELLS;i++){
    for(int j=0;j<(int)Tissue.cells[i].NV;j++){
      std::cout << Tissue.cells[i].Verticies[j][0] << "," <<
  Tissue.cells[i].Verticies[j][1] << std::endl;
    }
  }*/

  Tissue.CLEulerUpdate(1, 0.005);

  /*for(int i=0;i<(int)Tissue.NCELLS;i++){
    for(int j=0;j<(int)Tissue.cells[i].NV;j++){
      std::cout << Tissue.cells[i].Verticies[j][0] << "," <<
  Tissue.cells[i].Verticies[j][1] << std::endl;
    }
  }*/
  return 0;
}
