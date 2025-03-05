import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import plot
from matplotlib import pyplot as plt
from progressbar import progressbar

c = clDPM.Cell3D([0]*3,1.0,1.3) #input is [starting_point],CalA, radius
c2 = clDPM.Cell3D([0]*3,1.0,1.4)
c.Ka = 1.5
c.Kv = 0.6
c.Ks = 2.5
c2.Ka = 1.5
c2.Kv = 0.6
c2.Ks = 2.5

T = clDPM.Tissue3D([c2,c]*16,0.25) #input is: list of cell, and packing fraction (3D)
T.Kre = 10.0
T.Disperse2D()

Faces = T.Cells[0].GetFaces()
nsteps = 200
nout = 50
for i in progressbar(range(nout)):
  plot.PlotTissue2D(T)
  plt.savefig('/tmp/test_' + str(i) + '.png')
  plt.close()
  T.CLEulerUpdate(nsteps, 0.005)
