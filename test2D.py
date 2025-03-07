import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)
from plot import Plot2DTissue2D
from matplotlib import pyplot as plt
import numpy as np

import clDPM

c = clDPM.Cell2D(0,0,1.2,20,1.0) #input is startPoint.x startPoint.y, CalA, num verts, area
c2 = clDPM.Cell2D(0,0,1.2,18,1.3) #input is startPoint.x startPoint.y, CalA, num verts, area
c.Ka = 0.1
c.Kl = 1.0
c.Kb = 0.2
c2.Ka = 0.1
c2.Kl = 1.0
c2.Kb = 0.2

T = clDPM.Tissue2D([c,c2]*20,0.9); #input is list of cells and the initial packing fraction
T.Kre = 4
T.Disperse()

for i in range(50):
    Plot2DTissue2D(T)
    T.CLEulerUpdate(500,0.005);
    plt.savefig("/tmp/2d_"+str(i)+".png")
    plt.xlim(0,T.L)
    plt.ylim(0,T.L)
    plt.axis('equal')
    plt.close()
