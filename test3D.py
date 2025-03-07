import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import plot
from matplotlib import pyplot as plt
from progressbar import progressbar

c = clDPM.Cell3D([0]*3,1.05,1.1) #input is [starting_point],CalA, radius
c2 = clDPM.Cell3D([0]*3,1.05,1.4)
c.Ka = 0.6
c.Kv = 1.5
c.Ks = 5.0
c2.Ka = 0.6
c2.Kv = 1.5
c2.Ks = 5.0

T = clDPM.Tissue3D([c2,c]*21,0.35) #input is: list of cell, and packing fraction (3D)
T.Kre = 10.0
T.Disperse2D()

Faces = T.Cells[0].GetFaces()
nsteps = 500
nout = 1
for i in range(nout):
  print("Starting 3D simulation with "+str(T.NCELLS)+" particles for "+str(nsteps)+" timesteps...")
  T.CLEulerUpdate(nsteps, 0.001)
  print("Plotting figure...")
  plot.Plot3DTissue2D(T)
  filename = "test.png"
  plt.savefig(filename)
  print("figure saved to " + filename)
  plt.close()
