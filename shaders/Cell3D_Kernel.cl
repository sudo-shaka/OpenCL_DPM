
// Fixed OpenCL kernels and helpers
// Assumes OpenCL 1.2-ish vector functions; adjust if your platform differs.

constant uint NUM_FACES = 320;
constant uint NUM_VERTICES = 162;
constant float PI = 3.14159265358979323846f;

// --- Atomic float add (compare-and-swap) ---
void atomic_add_f(__global float* addr, float val) {
    union { unsigned int i; float f; } oldv, newv;
    unsigned int expected;
    do {
        // read current value
        oldv.f = *addr;
        newv.f = oldv.f + val;
        expected = oldv.i;
        // try to swap
    } while (atomic_cmpxchg((volatile __global int*)addr,
                            (int)expected,
                            (int)newv.i) != (int)expected);
}

// Atomic add for float3 (adds x,y,z,w components individually)
void atomic_add_float3(__global float3* addr, float3 val) {
    __global float* f = (__global float*)addr;
    atomic_add_f(&f[0], val.x);
    atomic_add_f(&f[1], val.y);
    atomic_add_f(&f[2], val.z);
}

// Simple center-of-mass for a cell (average of its vertices)
float3 GetCOM(__global const float3* Verts, int ci) {
    float3 sum = (float3)(0.0f,0.0f,0.0f);
    int base = ci * NUM_VERTICES;
    for (uint i=0; i<NUM_VERTICES; ++i) {
        sum += Verts[base + i];
    }
    float inv = 1.0f / (float)NUM_VERTICES;
    sum *= inv;
    return sum;
}

float getVolume(const __global uint3* VertIdxMat, __global float3* Verts, uint ci) {
    float volume = 0.0f;

    for (uint fi = 0; fi < NUM_FACES; fi++) {
        uint3 face = VertIdxMat[fi];
        int i0 = ci * NUM_VERTICES + face.x;
        int i1 = ci * NUM_VERTICES + face.y;
        int i2 = ci * NUM_VERTICES + face.z;

        float3 P0 = (float3)(Verts[i0].x, Verts[i0].y, Verts[i0].z);
        float3 P1 = (float3)(Verts[i1].x, Verts[i1].y, Verts[i1].z);
        float3 P2 = (float3)(Verts[i2].x, Verts[i2].y, Verts[i2].z);

        volume += dot(cross(P0, P1), P2) / 6.0f;
    }

    return fabs(volume);
}

__kernel void VolumeForceUpdate(
    __global const uint3* VertIdxMat,
    __global float3* Verts,
    __global float3* Forces,
    int NCELLS,
    __global float* Kv,
    __global float* v0,
    __global float* cellVolumes 
) {
    uint fi = get_global_id(0);
    uint ci = get_global_id(1);

    if (ci >= NCELLS || fi >= NUM_FACES || Kv[ci] == 0.0f) return;

    // Compute volume ONCE per cell (thread fi == 0)
    if (fi == 0) {
        cellVolumes[ci] = getVolume(VertIdxMat, Verts, ci);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    float volume = cellVolumes[ci];
    float volumeStrain = (volume / v0[ci]) - 1.0f;

    uint3 face = VertIdxMat[fi];
    int i0 = ci * NUM_VERTICES + face.x;
    int i1 = ci * NUM_VERTICES + face.y;
    int i2 = ci * NUM_VERTICES + face.z;

    float3 P0 = Verts[i0];
    float3 P1 = Verts[i1];
    float3 P2 = Verts[i2];

    float3 COM = GetCOM(Verts, ci);
    float3 A = P1 - COM;
    float3 B = P2 - COM;
    float3 C = P0 - COM;

    float3 grad0 = cross(A, B);
    float3 grad1 = cross(B, C);
    float3 grad2 = cross(C, A);

    float Kv_val = Kv[ci];
    float3 f0 = -Kv_val * volumeStrain * grad0 / 6.0f;
    float3 f1 = -Kv_val * volumeStrain * grad1 / 6.0f;
    float3 f2 = -Kv_val * volumeStrain * grad2 / 6.0f;

    atomic_add_float3(&Forces[i0], f0);
    atomic_add_float3(&Forces[i1], f1);
    atomic_add_float3(&Forces[i2], f2);
}

__kernel void SurfaceAreaForceUpdate(
    __global const uint3* VertIdxMat,
    __global float3* Verts,
    __global float3* Forces,
    uint NCELLS,
    __global float* Ka,   // surface stiffness per cell
    __global float* l0    // desired bond length per cell
) {
    uint fi = get_global_id(0);  // face index
    uint ci = get_global_id(1);  // cell index

    if (ci >= NCELLS || fi >= NUM_FACES) return;
    if (Ka[ci] <= 1e-6f) return;

    uint3 face = VertIdxMat[fi];
    int i0 = ci * NUM_VERTICES + face.x;
    int i1 = ci * NUM_VERTICES + face.y;
    int i2 = ci * NUM_VERTICES + face.z;

    float3 P0 = Verts[i0];
    float3 P1 = Verts[i1];
    float3 P2 = Verts[i2];

    // edge vectors
    float3 e0 = P1 - P0; // edge (0->1)
    float3 e1 = P2 - P1; // edge (1->2)
    float3 e2 = P0 - P2; // edge (2->0)

    // edge lengths
    float len0 = length(e0);
    float len1 = length(e1);
    float len2 = length(e2);

    // normalized directions
    float3 u0 = e0 / len0;
    float3 u1 = e1 / len1;
    float3 u2 = e2 / len2;

    float target = l0[ci];
    float stiffness = Ka[ci];

    // strains
    float dl0 = (len0 / target) - 1.0f;
    float dl1 = (len1 / target) - 1.0f;
    float dl2 = (len2 / target) - 1.0f;

    // edge spring forces: F = -Ka * strain * direction
    float3 f01 = -stiffness * dl0 * u0; // on vertex 0, toward vertex 1
    float3 f12 = -stiffness * dl1 * u1; // on vertex 1, toward vertex 2
    float3 f20 = -stiffness * dl2 * u2; // on vertex 2, toward vertex 0

    // apply equal and opposite forces on connected vertices
    // edge 0\u20131
    atomic_add_float3(&Forces[i0],  f01);
    atomic_add_float3(&Forces[i1], -f01);

    // edge 1\u20132
    atomic_add_float3(&Forces[i1],  f12);
    atomic_add_float3(&Forces[i2], -f12);

    // edge 2\u20130
    atomic_add_float3(&Forces[i2],  f20);
    atomic_add_float3(&Forces[i0], -f20);
}


// ---------------- StickToSurface ----------------
__kernel void StickToSurface(__global const uint3* VertIdxMat,
                             __global const float3* Verts,
                             __global float3* Forces,
                             uint NCELLS,
                             __global const float* Ks,
                             __global const float* l0)
{
    uint ci = get_global_id(1);
    uint fi = get_global_id(0);

    if (ci >= NCELLS) return;
    if (fi >= NUM_FACES) return;
    if (Ks[ci] == 0.0f) return;

    uint3 face = VertIdxMat[fi];
    int base = ci * NUM_VERTICES;
    int vi0 = base + face.x;
    int vi1 = base + face.y;
    int vi2 = base + face.z;

    float3 pos0 = Verts[vi0];
    float3 pos1 = Verts[vi1];
    float3 pos2 = Verts[vi2];

    float3 A = (float3)(pos1.x - pos0.x, pos1.y - pos0.y, pos1.z - pos0.z);
    float3 B = (float3)(pos2.x - pos0.x, pos2.y - pos0.y, pos2.z - pos0.z);

    float3 normal = cross(A, B);
    bool isUpward = normal.z > 0.0f;
    if (!isUpward) return;

    float ks = Ks[ci];
    float L0 = l0[ci];
    if (L0 == 0.0f) L0 = 1e-12f;

    // apply vertical (z) corrective forces if below z<0
    if (pos0.z < 0.0f) {
        float dz = pos0.z;
        atomic_add_f(&((__global float*)&Forces[vi0])[2], -ks * (dz / L0) * (1.0f/3.0f));
    }
    if (pos1.z < 0.0f) {
        float dz = pos1.z;
        atomic_add_f(&((__global float*)&Forces[vi1])[2], -ks * (dz / L0) * (1.0f/3.0f));
    }
    if (pos2.z < 0.0f) {
        float dz = pos2.z;
        atomic_add_f(&((__global float*)&Forces[vi2])[2], -ks * (dz / L0) * (1.0f/3.0f));
    }

    // push towards COM to stick
    float3 COM = GetCOM(Verts, ci);
    float3 dir0 = normalize((float3)(pos0.x-COM.x, pos0.y-COM.y, pos0.z-COM.z));
    float3 dir1 = normalize((float3)(pos1.x-COM.x, pos1.y-COM.y, pos1.z-COM.z));
    float3 dir2 = normalize((float3)(pos2.x-COM.x, pos2.y-COM.y, pos2.z-COM.z));

    float3 add0 = (float3)(0.3f * ks * dir0.x, 0.3f * ks * dir0.y, 0.3f * ks * dir0.z);
    float3 add1 = (float3)(0.3f * ks * dir1.x, 0.3f * ks * dir1.y, 0.3f * ks * dir1.z);
    float3 add2 = (float3)(0.3f * ks * dir2.x, 0.3f * ks * dir2.y, 0.3f * ks * dir2.z);

    atomic_add_float3(&Forces[vi0], add0);
    atomic_add_float3(&Forces[vi1], add1);
    atomic_add_float3(&Forces[vi2], add2);
}

// ---------------- Repelling Forces (winding number based) ----------------
__kernel void RepellingForces(__global const uint3* VertIdxMat,
                              __global const float3* Verts,
                              __global float3* Forces,
                              uint NCELLS,
                              __global const float* l0,
                              float Kc,
                              int PBC,
                              float L)
{
    uint ci = get_global_id(1);
    uint vi = get_global_id(0);

    if (ci >= NCELLS) return;
    if (vi >= NCELLS * NUM_VERTICES) return; // vi is a global vertex index across all cells?
    // NOTE: original code used vi as global vertex index; ensure dispatch matches this expectation

    if (Kc == 0.0f) return;

    // convert vertex index to cell/vertex if needed
    // here we interpret vi as the absolute index of vertex across all cells
    float3 point = Verts[vi];
    // find owning cell of this vertex
    int ci_from_vi = vi / NUM_VERTICES;

    float3 COMi = GetCOM(Verts, ci_from_vi);
for(int cj=0; cj<NCELLS; cj++){
    if(cj == ci) continue;

    float totalOmega = 0.0f;
    float3 COMJ = GetCOM(Verts,cj);

    // PBC shift
    float3 shift = (float3)(0,0,0);
    if(PBC){
        shift = L * round((COMi - COMJ)/L);
        COMJ += shift;
    }

    for(uint fj=0; fj<NUM_FACES; fj++){
        uint3 vj = VertIdxMat[fj];
        float3 a = Verts[cj*NUM_VERTICES + vj.x] - Verts[vi] + shift;
        float3 b = Verts[cj*NUM_VERTICES + vj.y] - Verts[vi] + shift;
        float3 c = Verts[cj*NUM_VERTICES + vj.z] - Verts[vi] + shift;

        float3 u = normalize(a);
        float3 v = normalize(b);
        float3 w = normalize(c);

        float denom = 1.0f + dot(u,v) + dot(v,w) + dot(w,u);
        if(denom < 1e-8f) continue;
        float num = dot(u, cross(v,w));
        float omega = 2.0f * atan2(num, denom);
        totalOmega += omega;
    }

    float windingNumber = totalOmega / (4.0f * M_PI);
    if(fabs(windingNumber) < 1e-6f) continue;

    float3 dir = normalize(Verts[vi] - COMJ); // CORRECT: away from other cell
    float3 tmpForce = fabs(windingNumber) * 0.5f * Kc * dir;
    atomic_add_float3(&Forces[vi], tmpForce);
}


}

// ---------------- All Vertex Attraction ----------------
__kernel void AllVertAttraction(__global const float3* Verts,
                                __global float3* Forces,
                                __global const float* l0,
                                float L,
                                int NCELLS,
                                int PBC,
                                float Kat)
{
    if (Kat == 0.0f) return;

    uint ci = get_global_id(1);
    uint vi = get_global_id(0);

    if (ci >= (uint)NCELLS) return;
    if (vi >= NUM_VERTICES) return;

    int base = ci * NUM_VERTICES;
    int vert_index = base + vi;
    float3 p1 = Verts[vert_index];

    for (int cj = 0; cj < NCELLS; ++cj) {
        if (cj == (int)ci) continue;
        int basej = cj * NUM_VERTICES;
        for (int vj = 0; vj < (int)NUM_VERTICES; ++vj) {
            float3 p2 = Verts[basej + vj];
            float3 delta4 = p2 - p1;
            float3 delta = (float3)(delta4.x, delta4.y, delta4.z);

            if (PBC) {
                float rx = round(delta.x / L);
                float ry = round(delta.y / L);
                float rz = round(delta.z / L);
                delta -= (float3)(L*rx, L*ry, L*rz);
            }

            float dist = length(delta);
            float L0 = l0[ci];
            if (L0 < 1e-12f) L0 = 1e-12f;

            if (dist < L0 * 2.0f && dist > 1e-12f) {
                float3 tmpForce = (float3)(Kat * 0.5f * (dist / L0 - 1.0f) * (delta.x / dist),
                                           Kat * 0.5f * (dist / L0 - 1.0f) * (delta.y / dist),
                                           Kat * 0.5f * (dist / L0 - 1.0f) * (delta.z / dist));
                atomic_add_float3(&Forces[vert_index], (float3)(-tmpForce.x, -tmpForce.y, -tmpForce.z));
                atomic_add_float3(&Forces[basej + vj], tmpForce);
            }
        }
    }
}

// ---------------- Euler Position update (explicit integrator) ----------------
__kernel void EulerPosition(__global float3* Verts, __global const float3* Forces, float dt, uint NCELLS) {
    uint ci = get_global_id(1);
    uint vi = get_global_id(0);

    if (ci >= NCELLS) return;
    if (vi >= NUM_VERTICES) return;

    uint vert_index = ci * NUM_VERTICES + vi;
    // simple Euler: x += F * dt
    Verts[vert_index] += Forces[vert_index] * dt;
}
