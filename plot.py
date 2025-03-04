import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import numpy as np
import matplotlib.pyplot as plt

def PlotTissue3D(T):
  Faces = T.Cells[0].GetFaces()
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(projection='3d')
  for ci in range(T.NCELLS):
    pos = T.Cells[ci].GetPositions()
    if(np.isnan(pos).any()):
      print(" ERROR! NaN values in position data. Exiting...")
      exit(0)
    x, y, z = np.mod(pos[0],T.L), np.mod(pos[1],T.L), np.mod(pos[2],T.L)
    for face in Faces:
      vx, vy, vz = [x[i] for i in face],[y[i] for i in face],[z[i] for i in face]
      if max(vx) - min(vx) <= T.L/2 and max(vy) - min(vy) <= T.L/2 and max(vz) - min(vz) <= T.L/2:
        ax.plot(vx, vy ,vz, 'k-',alpha=0.6)  # Close the loop by adding the first vertex at the end
    ax.scatter(x, y, z)

  ax.set_xlim(0,T.L)
  ax.set_ylim(0,T.L)
  ax.set_zlim(0,T.L)


def PlotTissue2D(T):
  Faces = T.Cells[0].GetFaces()
  for ci in range(T.NCELLS):
    pos = T.Cells[ci].GetPositions()
    if(np.isnan(pos).any()):
      print(" ERROR! NaN values in position data. Exiting...")
      exit(0)
    x, y= np.mod(pos[0],T.L), np.mod(pos[1],T.L)
    for face in Faces:
      vx, vy= [x[i] for i in face],[y[i] for i in face]

      if max(vx) - min(vx) <= T.L/2 and max(vy) - min(vy) <= T.L/2:
        plt.plot(vx, vy, 'k-',alpha=0.6)  # Close the loop by adding the first vertex at the end
    plt.scatter(x, y)

  plt.xlim(0,T.L)
  plt.ylim(0,T.L)
