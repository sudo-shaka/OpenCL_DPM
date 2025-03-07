import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)
from plot import Plot2DTissue2D
from matplotlib import pyplot as plt
from progressbar import progressbar

import clDPM

c = clDPM.Cell2D(0,0,1.2,25,1.0) #input is startPoint.x startPoint.y, CalA, num verts, area
c2 = clDPM.Cell2D(0,0,1.2,22,1.3) #input is startPoint.x startPoint.y, CalA, num verts, area
c.Ka = 0.1
c.Kl = 1.0
c.Kb = 0.05
c2.Ka = 0.1
c2.Kl = 1.0
c2.Kb = 0.05

T = clDPM.Tissue2D([c,c2]*200,0.9); #input is list of cells and the initial packing fraction
T.Kre = 1.0
T.Kat = 0.5
T.Disperse()

for i in progressbar(range(50)):
    Plot2DTissue2D(T)
    T.CLEulerUpdate(500,0.005);
    plt.savefig("/tmp/2d_"+str(i)+".png")
    plt.xlim(0,T.L)
    plt.ylim(0,T.L)
    plt.axis('equal')
    plt.close()

print("Images saved to /tmp/*.png")
