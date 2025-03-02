#include  <vector>
#include <string>
#include "cell.hpp"

#ifndef __TISSUE__
#define __TISSUE__

namespace DPM{
  class Tissue3D{
    public:
      int NCELLS;
      int PBC; //(boolean) periodic boundary conditions
      float L;
      float Kre;
      float Kat;
      std::string attractionMethod;
      std::vector<DPM::Cell3D> Cells;

      Tissue3D(std::vector<DPM::Cell3D> cells, float phi0);
      
      void CLEulerUpdate(int nsteps,  float dt);
      void Disperse2D();
      void Disperse3D();
      std::vector<std::vector<float>> GetVesselPositions(int ci);
  };

  class Tissue2D{
    public:
      int NCELLS;
      bool PBC;
      float Kre;
      float Kat;

      Tissue2D(std::vector<DPM::Cell2D> cells, float  phi0);

      void Disperse();
      void  CLEulerUpdate();
  };
}

#endif
