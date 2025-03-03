import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import numpy as np
from matplotlib import pyplot as plt
from progressbar import progressbar

c = clDPM.Cell3D([0,0,0],1.0,1.2);
c2 = clDPM.Cell3D([0,0,0],1.05,1.5)
c.Ks = 1.5
c.Ka = 2.0
c.Kv = 0.7
c2.Ka = 2.0
c2.Ks = 1.5
c2.Kv = 0.7

T = clDPM.Tissue3D([c,c2]*5,.25)
T.Kre = 50.0
T.Disperse2D()

Faces = T.Cells[0].GetFaces()
nsteps = 50
nout = 50
for i in progressbar(range(nout)):
  plt.figure(figsize=(10,10))
  for ci in range(len(T.Cells)):
    pos = T.Cells[ci].GetPositions()
    x, y = np.mod(pos[0], T.L), np.mod(pos[1], T.L)
    # Plot face edges
    for face in Faces:
      vx = [x[f] for f in face]
      vy = [y[f] for f in face]
      # Check if any point is on the other side of the boundary condition
      if max(vx) - min(vx) <= T.L/2 and max(vy) - min(vy) <= T.L/2:
        plt.plot(vx, vy, 'k-')  # Close the loop by adding the first vertex at the end
    plt.scatter(x, y)

  plt.xlim(0, T.L)
  plt.ylim(0, T.L)
  plt.savefig('/tmp/test_' + str(i) + '.png')
  T.CLEulerUpdate(nsteps, 0.005)
