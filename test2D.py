import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)
from matplotlib import pyplot as plt
import numpy as np

import clDPM

c = clDPM.Cell2D(0,0,1.2,50,1.0) #input is startPoint.x startPoint.y, CalA, num verts, area
c2 = clDPM.Cell2D(0,0,1.2,55,1.3) #input is startPoint.x startPoint.y, CalA, num verts, area
c.Ka = 5.0
c.Kl = 1.0
c.Kb = 0.2
c2.Ka = 5.0
c2.Kl = 1.0
c2.Kb = 0.2

T = clDPM.Tissue2D([c,c2]*20,0.75);
T.Kre = 4
T.Disperse()

for i in range(50):
    T.CLEulerUpdate(200,0.001);
    plt.figure(figsize=(10,10))
    for ci in range(T.NCELLS):
        pos=np.array(T.Cells[ci].Verts);
        pos=np.mod(pos.T,T.L)
        plt.scatter(pos[0],pos[1])
    plt.savefig("/tmp/2d_"+str(i)+".png")
    plt.xlim(0,T.L)
    plt.ylim(0,T.L)
    plt.axis('equal')
    plt.close()
