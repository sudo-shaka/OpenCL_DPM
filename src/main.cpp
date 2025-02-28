#include"cell.hpp"

int main(){
  std::array<float,3> startp = {0,0,0};
  DPM::Cell3D Cell(startp, 1.05, 5.0);
  Cell.Kv = 1.0;
  Cell.Ka = 1.0;
  Cell.Kb = 0.05;
  Cell.CLShapeEuler(10000, 0.001);

  return 0;
}
