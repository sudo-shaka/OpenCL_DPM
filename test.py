import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM;
from matplotlib import pyplot as plt

c = clDPM.Cell3D([0,0,0],1.0,1.5);
c.Ks = 1.0
c.Ka = 2.0
c.Kv = 1.0

T = clDPM.Tissue3D([c]*30, 0.35)
T.Kre = 1.0
T.Disperse2D()
T.CLEulerUpdate(1000,0.005);


for ci in range(len(T.Cells)):
    pos = T.Cells[ci].GetPositions();
    plt.scatter(pos[0],pos[1]);
plt.axis('equal')
plt.show()
