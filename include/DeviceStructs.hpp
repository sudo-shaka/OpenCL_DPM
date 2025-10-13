#ifndef __DEVPOD__
#define __DEVPOD__

struct DeviceVertex {
  float X, Y, Z;
  float Fx, Fy, Fz;
};

struct DeviceCell3DData {
  int NV;
  int nTriangles;
  float Kv, Ka, Ks;
  float v0, a0, l0;
  float idealForce;
  float COMX, COMY, COMZ;
  float Volume;
};

#endif