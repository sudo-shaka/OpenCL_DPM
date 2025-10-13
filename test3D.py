import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import plot
from matplotlib import pyplot as plt
from progressbar import progressbar

c = clDPM.Cell3D([0]*3,1.0,1.1) #input is [starting_point],CalA, radius
c.Ka = 2.0
c.Kv = 5.0
c.Ks = 10.0

T = clDPM.Tissue3D([c]*32,0.5) #input is: list of cell, and packing fraction (3D)
T.Kre = 25.0
T.Disperse2D()

Faces = T.Cells[0].GetFaces()
nsteps = 25
nout = 10
print("Starting 3D simulation with "+str(T.NCELLS)+" particles for "+str(nsteps)+" timesteps " + str(nout) + " times")
for i in progressbar(range(nout)):
  T.CLEulerUpdate(nsteps, 0.001)
  plot.Plot3DTissue2D(T)
  filename = "test"+str(i)+".png"
  plt.savefig(filename)
  plt.close()
