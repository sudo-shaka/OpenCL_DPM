import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import plot
from matplotlib import pyplot as plt
from progressbar import progressbar

c = clDPM.Cell3D([0]*3,1.0,1.1) #input is [starting_point],CalA, radius
c2 = clDPM.Cell3D([0]*3,1.0,1.4)
c.Ka = 0.0
c.Kv = 0.0
c.Ks = 0.0
c2.Ka = 0.0
c2.Kv = 0.0
c2.Ks = 0.0

T = clDPM.Tissue3D([c2,c]*16,0.25) #input is: list of cell, and packing fraction (3D)
T.Kre = 0.0
T.Kat = 0.0
T.Disperse2D()

Faces = T.Cells[0].GetFaces()
nsteps = 1
nout = 20
print("Starting 3D simulation with "+str(T.NCELLS)+" particles for "+str(nsteps)+" timesteps " + str(nout) + " times")
for i in progressbar(range(nout)):
  T.CLEulerUpdate(nsteps, 0.005)
  plot.Plot3DTissue2D(T)
  filename = "test"+str(i)+".png"
  plt.savefig(filename)
  plt.close()
print("figure saved to " + filename)