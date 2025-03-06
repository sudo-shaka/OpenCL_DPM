import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import plot
from matplotlib import pyplot as plt
from progressbar import progressbar

c = clDPM.Cell3D([0]*3,1.05,1.1) #input is [starting_point],CalA, radius
c2 = clDPM.Cell3D([0]*3,1.05,1.4)
c.Ka = 1.5
c.Kv = 0.6
c.Ks = 4.0
c2.Ka = 1.5
c2.Kv = 0.6
c2.Ks = 4.0

T = clDPM.Tissue3D([c2,c]*50,0.3) #input is: list of cell, and packing fraction (3D)
T.Kre = 10.0
T.Disperse2D()

Faces = T.Cells[0].GetFaces()
nsteps = 100
nout = 10
for i in progressbar(range(nout)):
  plot.Plot3DTissue2D(T)
  plt.savefig('/tmp/test_' + str(i) + '.png')
  plt.close()
  T.CLEulerUpdate(nsteps, 0.001)
