import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)
from matplotlib import pyplot as plt
import numpy as np

import clDPM

c = clDPM.Cell2D(0,0,1.2,40,1.1) #input is startPoint.x startPoint.y, CalA, num verts, area
c.Ka = 1.0
c.Kl = 1.0
c.Kb = 0.1

T = clDPM.Tissue2D([c]*30,0.85);
T.Disperse()
T.CLEulerUpdate(100,0.005);

pos=np.array(T.Cells[0].Verts);
pos = pos.T
print(pos)
plt.scatter(pos[0],pos[1])
plt.savefig('2dtest.png')
