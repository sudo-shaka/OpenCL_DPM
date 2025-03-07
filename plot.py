import ctypes
ctypes.CDLL("libOpenCL.so", mode=ctypes.RTLD_GLOBAL)

import numpy as np
import matplotlib.pyplot as plt

def Plot2DTissue2D(T):
    plt.figure(figsize=(10,10))
    for ci in range(T.NCELLS):
        pos=np.array(T.Cells[ci].Verts);
        pos=np.mod(pos.T,T.L)
        force = np.array(T.Cells[ci].Forces)
        force = force.T
        force = force[0] + force[1]
        plt.scatter(pos[0],pos[1],c=force,cmap='coolwarm')
        np.append(pos[0],pos[0][0])
        np.append(pos[1],pos[1][0])
        if np.max(pos[0]) - np.min(pos[0]) > T.L/2 or np.max(pos[1]) - np.min(pos[1]) > T.L/2:
          continue
        plt.plot(pos[0],pos[1],'-k')
    plt.axis('equal')

def Plot3DTissue3D(T):
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
        ax.plot(vx, vy ,vz,color='gray',alpha=0.6)  # Close the loop by adding the first vertex at the end
    ax.scatter(x, y, z,alpha=1)

  ax.set_xlim(0,T.L)
  ax.set_ylim(0,T.L)
  ax.set_zlim(0,T.L)


def Plot3DTissue2D(T):
  Faces = T.Cells[0].GetFaces()
  np.random.seed(1)
  r1 = np.random.rand(T.NCELLS)
  r2 = np.random.rand(T.NCELLS)
  r3 = np.random.rand(T.NCELLS)
  for ci in range(T.NCELLS):
    pos = T.Cells[ci].GetPositions()
    if(np.isnan(pos).any()):
      print(" ERROR! NaN values in position data. Exiting...")
      exit(0)
    x, y= np.mod(pos[0],T.L), np.mod(pos[1],T.L)
    for face in Faces:
      vx, vy= [x[i] for i in face],[y[i] for i in face]

      if max(vx) - min(vx) <= T.L/2 and max(vy) - min(vy) <= T.L/2:
        plt.plot(vx, vy,color=(r1[ci],r2[ci],r3[ci]))  # Close the loop by adding the first vertex at the end
    plt.scatter(x, y, s=3,color=(r1[ci],r2[ci],r3[ci]))
  plt.axis('equal')
  plt.xlim(0,T.L)
  plt.ylim(0,T.L)
