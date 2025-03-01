#include "Tissue.hpp"
#include "cell.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <thread>

namespace DPM{
  Tissue3D::Tissue3D(std::vector<DPM::Cell3D> cells,float phi0){
    Cells = cells;
    NCELLS  = Cells.size();
    float volume = 0.0f;
    for(int ci=0;ci<NCELLS;ci++){
      volume += Cells[ci].v0;
    }
    L = cbrt(volume)/phi0;
    PBC = true;
    attractionMethod.assign("General");
  }

  void Tissue3D::Disperse2D(){
    std::vector<float> X,Y,Fx,Fy;
    X.resize(NCELLS); Y.resize(NCELLS);
    Fx.resize(NCELLS); Fy.resize(NCELLS);
    float ri,rj,yi,yj,xi,xj,dx,dy,dist;
    float ux,uy,ftmp,fx,fy;
    int  i,j,count;
    for(i=0;i<NCELLS;i++){
        X[i] = drand48() * L;
        Y[i] = drand48() * L;
    }
    float oldU = 100, dU = 100;
    count = 0;
    while(dU > 1e-6){
        float  U = 0;
        for(i=0;i<NCELLS;i++){
            Fx[i] = 0.0;
            Fy[i] = 0.0;
        }

        for(i=0;i<NCELLS;i++){
            xi = X[i];
            yi = Y[i];
            ri = Cells[i].r0*2;
            for(j=0;j<NCELLS;j++){
                if(j != i){
                    xj = X[j];
                    yj = Y[j];
                    rj = Cells[j].r0*2;
                    dx = xj-xi;
                    dx -= L*round(dx/L);
                    dy = yj-yi;
                    dy -= L*round(dy/L);
                    dist = sqrt(dx*dx + dy*dy);
                    if(dist < 0.0)
                        dist *= -1;
                    if(dist <= (ri+rj)){
                        ux = dx/dist;
                        uy = dy/dist;
                        ftmp = (1.0-dist/(ri+rj))/(ri+rj);
                        fx = ftmp*ux;
                        fy = ftmp*uy;
                        Fx[i] -= fx;
                        Fy[i] -= fy;
                        Fy[j] += fy;
                        Fx[j] += fx;
                        U += 0.5*(1-(dist/(ri+rj))*(1-dist/(ri+rj)));
                    }
                }
            }
        }
        for(int i=0; i<NCELLS;i++){
            X[i] += 0.01*Fx[i];
            Y[i] += 0.01*Fy[i];
        }
        dU = U-oldU;
        if(dU < 0.0)
            dU *= -1;
        oldU = U;
        count++;
        if(count > 1e5){
            std::cerr << "Warning: Max timesteps for dispersion reached"  << std::endl;
            break;
        }
    }
    for(i=0; i<(int)NCELLS; i++){
      std::array<float,3> com = Cells[i].GetCOM();
        for(j=0;j<(int)Cells[i].NV;j++){
            Cells[i].Verts[j][0] -= com[0];
            Cells[i].Verts[j][1] -= com[1];
            Cells[i].Verts[j][0] -= X[i];
            Cells[i].Verts[j][1] -= Y[i];
        }
    }
  }
}
