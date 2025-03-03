import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM;
from matplotlib import pyplot as plt
import numpy as np

c = clDPM.Cell3D([0,0,0],1.0,1.5);
c.Ks = 1.0
c.Ka = 2.0
c.Kv = 1.0

T = clDPM.Tissue3D([c]*100, 0.2)
T.Kre = 1.0
T.Disperse2D()
T.CLEulerUpdate(100,0.005);

Faces = T.Cells[0].GetFaces()
plt.figure(figsize=(10,10))
for ci in range(len(T.Cells)):
    pos = T.Cells[ci].GetPositions()
    plt.scatter(pos[0],pos[1])
plt.xlim(0,T.L)
plt.xlim(0,T.L)
plt.axis('equal')
plt.savefig('test.png')
