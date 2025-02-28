#include"cell.hpp"
  
int main(){
  std::array<float,3> startp = {0,0,0};
  DPM::Cell3D Cell(startp, 1.05, 5.0);
  Cell.VolumeForceUpdate();
}
