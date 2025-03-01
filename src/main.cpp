#include <iostream>
#include "cell.hpp"

int main(){
  float r0 = 1.0f;
  std::array<float,3> startp = {0,0,r0};
  DPM::Cell3D Cell(startp, 1.2 ,r0);
  Cell.Kv = 1.0;
  Cell.Ka = 1.0;
  Cell.Ks = 0.0;


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
