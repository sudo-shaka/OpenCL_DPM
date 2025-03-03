import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM;

c = clDPM.Cell3D([0,0,0],1.0,1.5);

T = clDPM.Tissue3D([c]*100, 0.2)
T.CLEulerUpdate(100,0.0001);
