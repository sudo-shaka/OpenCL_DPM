import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import clDPM
import numpy as np
from matplotlib.pyplot as plt
from progressbar import progressbar

def PlotTissue2D(Tissue):
  Faces = T.Cells[0].GetFaces()
  plt.figure(figsize=(10,10))
  for ci in range(T.NCELLS):
    pos = T.Cells[ci].GetPositions()
    x, y = np.mod(pos[0],T.L), np.mod(pos[1],T.L)
    for face in Faces:
      vx, vy = [x[i] for i in face],[y[i] for i in face]

      if max(vx) - min(vx) <= T.L*0.8 and max(vy) - min(vy) <= T.L*0.8:
        plt.plot(vx, vy, 'k-')  # Close the loop by adding the first vertex at the end
    plt.scatter(x, y, s=3)

  plt.xlim(0,T.L)
  plt.ylim(0,T.L)
