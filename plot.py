import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import numpy as np
import matplotlib.pyplot as plt

def PlotTissue2D(T):
  Faces = T.Cells[0].GetFaces()
  plt.figure(figsize=(10,10))
  for ci in range(T.NCELLS):
    pos = T.Cells[ci].GetPositions()
    x, y = np.mod(pos[0],T.L), np.mod(pos[1],T.L)
    for face in Faces:
      vx, vy = [x[i] for i in face],[y[i] for i in face]

      if max(vx) - min(vx) <= T.L/2 and max(vy) - min(vy) <= T.L/2:
        plt.plot(vx, vy, 'k-',alpha=0.6)  # Close the loop by adding the first vertex at the end
    plt.scatter(x, y)

  plt.xlim(0,T.L)
  plt.ylim(0,T.L)
