import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
from plot import PlotTissue2D
from matplotlib import pyplot as plt
from progressbar import progressbar

c = clDPM.Cell3D([0,0,0],1.0,1.2) #input is [starting_point],CalA, radius
c2 = clDPM.Cell3D([0,0,0],1.05,0.1)
c.Ka = 2.0
c.Kv = 0.7
c.Ks = 2.5
c2.Ka = 2.0
c2.Kv = 0.7
c2.Ks = 2.5

T = clDPM.Tissue3D([c,c2]*50,.1) #input is: list of cell, and packing fraction (3D)
T.Kre = 50.0
T.Disperse2D()

Faces = T.Cells[0].GetFaces()
nsteps = 50
nout = 50
for i in progressbar(range(nout)):
  PlotTissue2D(T)
  plt.savefig('/tmp/test_' + str(i) + '.png')
  plt.close()
  T.CLEulerUpdate(nsteps, 0.005)
