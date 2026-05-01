/*LICENSE*/

#include <cstdio>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../grid/grid3D.h"
#include "RT_functions.h"
#include "radiation.h"
#ifdef MPI_CHOLLA
  #include "../mpi/mpi_routines.h"
#endif

#ifdef RT

__global__ void Load_RT_Buffer_kernel(int direction, int side, int size_buffer, 
                                      int n_i, int n_j, int nx, int ny, int nz,
                                      int n_ghost_transfer, int n_ghost_rt, int n_freq,
                                      struct Rad3D::RT_Fields rtFields, Real* transfer_buffer_d)
{
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_rf;
  tid   = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i * n_j);
  tid_j = (tid - tid_k * n_i * n_j) / n_i;
  tid_i = tid - tid_k * n_i * n_j - tid_j * n_i;

  // total number of cells in the rt grid
  int n_cells = nx * ny * nz;

  if (tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost_transfer) return;

  tid_buffer = tid_i + tid_j * n_i + tid_k * n_i * n_j;

  if (direction == 0) {
    if (side == 0) tid_rf = (n_ghost_rt + tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
    if (side == 1) tid_rf = (nx - n_ghost_rt - n_ghost_transfer + tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
  }
  if (direction == 1) {
    if (side == 0) tid_rf = (tid_i) + (n_ghost_rt + tid_k) * nx + (tid_j)*nx * ny;
    if (side == 1) tid_rf = (tid_i) + (ny - n_ghost_rt - n_ghost_transfer + tid_k) * nx + (tid_j)*nx * ny;
  }
  if (direction == 2) {
    if (side == 0) tid_rf = (tid_i) + (tid_j)*nx + (n_ghost_rt + tid_k) * nx * ny;
    if (side == 1) tid_rf = (tid_i) + (tid_j)*nx + (nz - n_ghost_rt - n_ghost_transfer + tid_k) * nx * ny;
  }


// rewrite once we're confident in the general sizing
#ifdef OTVET
  for (int i = 0; i < n_freq; i++) { // Suspicious -- valid only for OTVET
    transfer_buffer_d[tid_buffer + i * size_buffer]            = rtFields.dev_rf[tid_rf + (1 + i) * n_cells];
    transfer_buffer_d[tid_buffer + (n_freq + i) * size_buffer] = rtFields.dev_rf[tid_rf + (1 + n_freq + i) * n_cells];
  }
#endif //OTVET
#ifdef M1
  for (int i = 0; i < n_freq; i++) { // Suspicious -- BRANT  DEFINITELY NEEDS ALTERATION ERROR
    transfer_buffer_d[tid_buffer + i * size_buffer]            = rtFields.dev_rf[tid_rf + (1 + i) * n_cells];
    transfer_buffer_d[tid_buffer + (n_freq + i) * size_buffer] = rtFields.dev_rf[tid_rf + (1 + n_freq + i) * n_cells];
  }
#endif //M1
}

__global__ void Unload_RT_Buffer_kernel(int direction, int side, int size_buffer, int n_i, int n_j, int nx, int ny,
                                        int nz, int n_ghost_transfer, int n_ghost_rt, int n_freq,
                                        struct Rad3D::RT_Fields rtFields, Real* transfer_buffer_d)
{
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_buffer, tid_rf;
  tid   = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i * n_j);
  tid_j = (tid - tid_k * n_i * n_j) / n_i;
  tid_i = tid - tid_k * n_i * n_j - tid_j * n_i;

  // total number of cells in the rt grid
  int n_cells = nx * ny * nz;

  if (tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost_transfer) return;

  tid_buffer = tid_i + tid_j * n_i + tid_k * n_i * n_j;

  if (direction == 0) {
    if (side == 0) tid_rf = (n_ghost_rt - n_ghost_transfer + tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
    if (side == 1) tid_rf = (nx - n_ghost_rt + tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
  }
  if (direction == 1) {
    if (side == 0) tid_rf = (tid_i) + (n_ghost_rt - n_ghost_transfer + tid_k) * nx + (tid_j)*nx * ny;
    if (side == 1) tid_rf = (tid_i) + (ny - n_ghost_rt + tid_k) * nx + (tid_j)*nx * ny;
  }
  if (direction == 2) {
    if (side == 0) tid_rf = (tid_i) + (tid_j)*nx + (n_ghost_rt - n_ghost_transfer + tid_k) * nx * ny;
    if (side == 1) tid_rf = (tid_i) + (tid_j)*nx + (nz - n_ghost_rt + tid_k) * nx * ny;
  }


// update after we are confident in the bueffer sizing
#ifdef OTVET
  for (int i = 0; i < n_freq; i++) { // Suspicious -- valid only for OTVET
    rtFields.dev_rf[tid_rf + (1 + i) * n_cells]          = transfer_buffer_d[tid_buffer + i * size_buffer];
    rtFields.dev_rf[tid_rf + (1 + n_freq + i) * n_cells] = transfer_buffer_d[tid_buffer + (n_freq + i) * size_buffer];
  }
#endif //OTVET
#ifdef M1
  for (int i = 0; i < n_freq; i++) { // Suspicious -- BRANT  DEFINITELY NEEDS ALTERATION ERROR
    rtFields.dev_rf[tid_rf + (1 + i) * n_cells]          = transfer_buffer_d[tid_buffer + i * size_buffer];
    rtFields.dev_rf[tid_rf + (1 + n_freq + i) * n_cells] = transfer_buffer_d[tid_buffer + (n_freq + i) * size_buffer];
  }
#endif //M1

}

__global__ void Set_RT_Boundaries_Periodic_Kernel(int direction, int side, int n_i, int n_j, int nx, int ny, int nz,
                                                  int n_ghost, int n_freq, struct Rad3D::RT_Fields rtFields)
{
  int n_cells = nx * ny * nz;

  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_src, tid_dst;
  tid   = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i * n_j);
  tid_j = (tid - tid_k * n_i * n_j) / n_i;
  tid_i = tid - tid_k * n_i * n_j - tid_j * n_i;

  if (tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost) return;

  if (direction == 0) {
    if (side == 0) tid_src = (nx - 2 * n_ghost + tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
    if (side == 0) tid_dst = (tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
    if (side == 1) tid_src = (n_ghost + tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
    if (side == 1) tid_dst = (nx - n_ghost + tid_k) + (tid_i)*nx + (tid_j)*nx * ny;
  }
  if (direction == 1) {
    if (side == 0) tid_src = (tid_i) + (ny - 2 * n_ghost + tid_k) * nx + (tid_j)*nx * ny;
    if (side == 0) tid_dst = (tid_i) + (tid_k)*nx + (tid_j)*nx * ny;
    if (side == 1) tid_src = (tid_i) + (n_ghost + tid_k) * nx + (tid_j)*nx * ny;
    if (side == 1) tid_dst = (tid_i) + (ny - n_ghost + tid_k) * nx + (tid_j)*nx * ny;
  }
  if (direction == 2) {
    if (side == 0) tid_src = (tid_i) + (tid_j)*nx + (nz - 2 * n_ghost + tid_k) * nx * ny;
    if (side == 0) tid_dst = (tid_i) + (tid_j)*nx + (tid_k)*nx * ny;
    if (side == 1) tid_src = (tid_i) + (tid_j)*nx + (n_ghost + tid_k) * nx * ny;
    if (side == 1) tid_dst = (tid_i) + (tid_j)*nx + (nz - n_ghost + tid_k) * nx * ny;
  }

// update after we are confident in the bueffer sizing
#ifdef OTVET
  for (int i = 0; i < n_freq; i++) { // Suspicious -- valid only for OTVET
    rtFields.dev_rf[tid_dst + (1 + i) * n_cells]          = rtFields.dev_rf[tid_src + (1 + i) * n_cells];
    rtFields.dev_rf[tid_dst + (1 + n_freq + i) * n_cells] = rtFields.dev_rf[tid_src + (1 + n_freq + i) * n_cells];
  }
#endif //OTVET
#ifdef M1
  for (int i = 0; i < n_freq; i++) { // Suspicious -- BRANT  DEFINITELY NEEDS ALTERATION ERROR
    rtFields.dev_rf[tid_dst + (1 + i) * n_cells]          = rtFields.dev_rf[tid_src + (1 + i) * n_cells];
    rtFields.dev_rf[tid_dst + (1 + n_freq + i) * n_cells] = rtFields.dev_rf[tid_src + (1 + n_freq + i) * n_cells];
  }
#endif //M1
}

void __global__ Calc_Absorption_Kernel(int nx, int ny, int nz, Real dx, CrossSectionInCU xs,
                                       const Real* __restrict__ dens, Real* __restrict__ abc)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jk  = tid / nx;
  const int i   = tid % nx;
  const int j   = jk % ny;
  const int k   = jk / ny;

  if (k >= nz) return;

  const Real densHI   = dens[i + nx * (j + ny * (k + 0 * nz))];
  const Real densHeI  = dens[i + nx * (j + ny * (k + 2 * nz))];
  const Real densHeII = dens[i + nx * (j + ny * (k + 3 * nz))];

  abc[i + nx * (j + ny * (k + 0 * nz))] = dx * (xs.HIatHI * densHI);
  abc[i + nx * (j + ny * (k + 1 * nz))] = dx * (xs.HIatHeI * densHI + xs.HeIatHeI * densHeI);
  abc[i + nx * (j + ny * (k + 2 * nz))] =
      dx * (xs.HIatHeII * densHI + xs.HeIatHeII * densHeI + xs.HeIIatHeII * densHeII);
}

  #define PU(TT, I, J, K) rfNear[i + I + nx * (j + J + ny * (k + K))] * et##TT[i + I + nx * (j + J + ny * (k + K))]
  #define PV(TT, I, J, K) \
    rfFar[i + I + nx * (j + J + ny * (k + K))] * (1.0f / 3.0f)  // et for the far field is unitary matrix/3

void __global__ OTVETIteration_Kernel(int nx, int ny, int nz, int n_ghost, Real dx, bool lastIteration,
                                      const Real rsFarFactor, const Real* __restrict__ rs, const Real* __restrict__ et,
                                      const Real* __restrict__ rfOT, const Real* __restrict__ rfNear,
                                      const Real* __restrict__ rfFar, const Real* __restrict__ abc,
                                      Real* __restrict__ rfNearNew, Real* __restrict__ rfFarNew, int deb)
{
  const Real alpha     = 0.8;  // Parameters from cpp code
  const Real gamma     = 1;
  const Real epsNum    = 1.0e-6;
  const Real facOverOT = 2;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jk  = tid / nx;
  const int i   = tid % nx;
  const int j   = jk % ny;
  const int k   = jk / ny;

  if (i < n_ghost || j < n_ghost || k < n_ghost || i >= nx - n_ghost || j >= ny - n_ghost || k >= nz - n_ghost) return;

  const int fieldPitch = nx * ny * nz;

  //
  //  Set pointers into et array pointing to specific fields.
  //  Names are the same as in cpp code
  //
  const Real* etXX = et + 0 * fieldPitch;
  const Real* etXY = et + 1 * fieldPitch;
  const Real* etYY = et + 2 * fieldPitch;
  const Real* etXZ = et + 3 * fieldPitch;
  const Real* etYZ = et + 4 * fieldPitch;
  const Real* etZZ = et + 5 * fieldPitch;

  //
  //  Compute edge projections U^(x,y,z)_{i,j,k}
  //

  //
  //  X-direction
  //
  // float ahx = epsNum + 0.5f*(abc.Val(0,i,j,k)+abc.Val(0,i+1,j,k));
  const Real ahpcc = epsNum + 0.5f * (abc[i + nx * (j + ny * k)] + abc[i + 1 + nx * (j + ny * k)]);
  const Real ahmcc = epsNum + 0.5f * (abc[i + nx * (j + ny * k)] + abc[i - 1 + nx * (j + ny * k)]);

  // float ux = rfNear.Val(0,i+1,j,k)*et.Val(0,i+1,j,k) - rfNear.Val(0,i,j,k)*et.Val(0,i,j,k);
  Real uxp = PU(XX, 1, 0, 0) - PU(XX, 0, 0, 0);
  Real uxm = PU(XX, 0, 0, 0) - PU(XX, -1, 0, 0);
  Real vxp = PV(XX, 1, 0, 0) - PV(XX, 0, 0, 0);
  Real vxm = PV(XX, 0, 0, 0) - PV(XX, -1, 0, 0);

  // float uy = 0.25f*(rfNear.Val(0,i+1,j+1,k)*et.Val(1,i+1,j+1,k) + rfNear.Val(0,i,j+1,k)*et.Val(1,i,j+1,k) -
  //                   rfNear.Val(0,i+1,j-1,k)*et.Val(1,i+1,j-1,k) - rfNear.Val(0,i,j-1,k)*et.Val(1,i,j-1,k));
  Real uyp = 0.25f * (PU(XY, 1, 1, 0) + PU(XY, 0, 1, 0) - PU(XY, 1, -1, 0) - PU(XY, 0, -1, 0));
  Real uym = 0.25f * (PU(XY, 0, 1, 0) + PU(XY, -1, 1, 0) - PU(XY, 0, -1, 0) - PU(XY, -1, -1, 0));
  Real vyp = 0;
  Real vym = 0;

  // float uz = 0.25f*(rfNear.Val(0,i+1,j,k+1)*et.Val(3,i+1,j,k+1) + rfNear.Val(0,i,j,k+1)*et.Val(3,i,j,k+1) -
  //                   rfNear.Val(0,i+1,j,k-1)*et.Val(3,i+1,j,k-1) - rfNear.Val(0,i,j,k-1)*et.Val(3,i,j,k-1));
  Real uzp = 0.25f * (PU(XZ, 1, 0, 1) + PU(XZ, 0, 0, 1) - PU(XZ, 1, 0, -1) - PU(XZ, 0, 0, -1));
  Real uzm = 0.25f * (PU(XZ, 0, 0, 1) + PU(XZ, -1, 0, 1) - PU(XZ, 0, 0, -1) - PU(XZ, -1, 0, -1));
  Real vzp = 0;
  Real vzm = 0;

  // flux.Val(0,i,j,k) = (ux+uy+uz)/ahx;
  const Real fuxp = (uxp + uyp + uzp) / ahpcc;
  const Real fuxm = (uxm + uym + uzm) / ahmcc;
  const Real fvxp = (vxp + vyp + vzp) / ahpcc;
  const Real fvxm = (vxm + vym + vzm) / ahmcc;

  //
  //  Y-direction
  //
  // float ahy = epsNum + 0.5f*(abc.Val(0,i,j,k)+abc.Val(0,i,j+1,k));
  const Real ahcpc = epsNum + 0.5f * (abc[i + nx * (j + ny * k)] + abc[i + nx * (j + 1 + ny * k)]);
  const Real ahcmc = epsNum + 0.5f * (abc[i + nx * (j + ny * k)] + abc[i + nx * (j - 1 + ny * k)]);

  // float uy = rfNear.Val(0,i,j+1,k)*et.Val(2,i,j+1,k) - rfNear.Val(0,i,j,k)*et.Val(2,i,j,k);
  uyp = PU(YY, 0, 1, 0) - PU(YY, 0, 0, 0);
  uym = PU(YY, 0, 0, 0) - PU(YY, 0, -1, 0);
  vyp = PV(YY, 0, 1, 0) - PV(YY, 0, 0, 0);
  vym = PV(YY, 0, 0, 0) - PV(YY, 0, -1, 0);

  // float ux = 0.25f*(rfNear.Val(0,i+1,j+1,k)*et.Val(1,i+1,j+1,k) + rfNear.Val(0,i+1,j,k)*et.Val(1,i+1,j,k) -
  //                   rfNear.Val(0,i-1,j+1,k)*et.Val(1,i-1,j+1,k) - rfNear.Val(0,i-1,j,k)*et.Val(1,i-1,j,k));
  uxp = 0.25f * (PU(XY, 1, 1, 0) + PU(XY, 1, 0, 0) - PU(XY, -1, 1, 0) - PU(XY, -1, 0, 0));
  uxm = 0.25f * (PU(XY, 1, 0, 0) + PU(XY, 1, -1, 0) - PU(XY, -1, 0, 0) - PU(XY, -1, -1, 0));
  vxp = 0;
  vxm = 0;

  // float uz = 0.25f*(rfNear.Val(0,i,j+1,k+1)*et.Val(4,i,j+1,k+1) + rfNear.Val(0,i,j,k+1)*et.Val(4,i,j,k+1) -
  //                   rfNear.Val(0,i,j+1,k-1)*et.Val(4,i,j+1,k-1) - rfNear.Val(0,i,j,k-1)*et.Val(4,i,j,k-1));
  uzp = 0.25f * (PU(YZ, 0, 1, 1) + PU(YZ, 0, 0, 1) - PU(YZ, 0, 1, -1) - PU(YZ, 0, 0, -1));
  uzm = 0.25f * (PU(YZ, 0, 0, 1) + PU(YZ, 0, -1, 1) - PU(YZ, 0, 0, -1) - PU(YZ, 0, -1, -1));
  vzp = 0;
  vzm = 0;

  // flux.Val(1,i,j,k) = (ux+uy+uz)/ahy;
  const Real fuyp = (uxp + uyp + uzp) / ahcpc;
  const Real fuym = (uxm + uym + uzm) / ahcmc;
  const Real fvyp = (vxp + vyp + vzp) / ahcpc;
  const Real fvym = (vxm + vym + vzm) / ahcmc;

  //
  //  Z-direction
  //
  // float ahz = epsNum + 0.5f*(abc.Val(0,i,j,k)+abc.Val(0,i,j,k+1));
  const Real ahccp = epsNum + 0.5f * (abc[i + nx * (j + ny * k)] + abc[i + nx * (j + ny * (k + 1))]);
  const Real ahccm = epsNum + 0.5f * (abc[i + nx * (j + ny * k)] + abc[i + nx * (j + ny * (k - 1))]);

  // float uz = rfNear.Val(0,i,j,k+1)*et.Val(5,i,j,k+1) - rfNear.Val(0,i,j,k)*et.Val(5,i,j,k);
  uzp = PU(ZZ, 0, 0, 1) - PU(ZZ, 0, 0, 0);
  uzm = PU(ZZ, 0, 0, 0) - PU(ZZ, 0, 0, -1);
  vzp = PV(ZZ, 0, 0, 1) - PV(ZZ, 0, 0, 0);
  vzm = PV(ZZ, 0, 0, 0) - PV(ZZ, 0, 0, -1);

  // float ux = 0.25f*(rfNear.Val(0,i+1,j,k+1)*et.Val(3,i+1,j,k+1) + rfNear.Val(0,i+1,j,k)*et.Val(3,i+1,j,k) -
  //                   rfNear.Val(0,i-1,j,k+1)*et.Val(3,i-1,j,k+1) - rfNear.Val(0,i-1,j,k)*et.Val(3,i-1,j,k));
  uxp = 0.25f * (PU(XZ, 1, 0, 1) + PU(XZ, 1, 0, 0) - PU(XZ, -1, 0, 1) - PU(XZ, -1, 0, 0));
  uxm = 0.25f * (PU(XZ, 1, 0, 0) + PU(XZ, 1, 0, -1) - PU(XZ, -1, 0, 0) - PU(XZ, -1, 0, -1));
  vxp = 0;
  vxm = 0;

  // float uy = 0.25f*(rfNear.Val(0,i,j+1,k+1)*et.Val(4,i,j+1,k+1) + rfNear.Val(0,i,j+1,k)*et.Val(4,i,j+1,k) -
  //                   rfNear.Val(0,i,j-1,k+1)*et.Val(4,i,j-1,k+1) - rfNear.Val(0,i,j-1,k)*et.Val(4,i,j-1,k));
  uyp = 0.25f * (PU(YZ, 0, 1, 1) + PU(YZ, 0, 1, 0) - PU(YZ, 0, -1, 1) - PU(YZ, 0, -1, 0));
  uym = 0.25f * (PU(YZ, 0, 1, 0) + PU(YZ, 0, 1, -1) - PU(YZ, 0, -1, 0) - PU(YZ, 0, -1, -1));
  vyp = 0;
  vym = 0;

  // flux.Val(2,i,j,k) = (ux+uy+uz)/ahz;
  const Real fuzp = (uxp + uyp + uzp) / ahccp;
  const Real fuzm = (uxm + uym + uzm) / ahccm;
  const Real fvzp = (vxp + vyp + vzp) / ahccp;
  const Real fvzm = (vxm + vym + vzm) / ahccm;

  // float minus_wII = et.Val(0,iw,jw,kw)*(abch.Val(0,ihm,jw,kw)+abch.Val(0,ihp,jw,kw)) +
  // et.Val(2,iw,jw,kw)*(abch.Val(1,iw,jhm,kw)+abch.Val (1,iw,jhp,kw)) +
  // et.Val(5,iw,jw,kw)*(abch.Val(2,iw,jw,khm)+abch.Val(2,iw,jw,khp));
  const Real uminus_wII = etXX[i + nx * (j + ny * k)] * (1 / ahpcc + 1 / ahmcc) +
                          etYY[i + nx * (j + ny * k)] * (1 / ahcpc + 1 / ahcmc) +
                          etZZ[i + nx * (j + ny * k)] * (1 / ahccp + 1 / ahccm);
  const Real vminus_wII = (1.0f / 3.0f) / ahpcc + (1.0f / 3.0f) / ahmcc + (1.0f / 3.0f) / ahcpc +
                          (1.0f / 3.0f) / ahcmc + (1.0f / 3.0f) / ahccp + (1.0f / 3.0f) / ahccm;

  // float A = gamma/(1+gamma*(abc.Val(0,iw,jw,kw)+minus_wII));
  const Real Au = gamma / (1 + gamma * (abc[i + nx * (j + ny * k)] + uminus_wII));
  const Real Av = gamma / (1 + gamma * (abc[i + nx * (j + ny * k)] + vminus_wII));

  // float d = pars.dx*rs.Val(0,iw,jw,kw) - abc.Val(0,iw,jw,kw)*rfNear.Val(0,iw,jw,kw) + flux.Val(0,ihp,jw,kw) -
  // flux.Val(0,ihm,jw,kw) + flux.Val(1,iw,jhp,kw) - flux.Val(1,iw,jhm,kw) + flux.Val(2,iw,jw,khp) -
  // flux.Val(2,iw,jw,khm);
  const Real du = dx * rs[i + nx * (j + ny * k)] - abc[i + nx * (j + ny * k)] * rfNear[i + nx * (j + ny * k)] + fuxp -
                  fuxm + fuyp - fuym + fuzp - fuzm;
  const Real dv = dx * rsFarFactor * rfFar[i + nx * (j + ny * k)] -
                  abc[i + nx * (j + ny * k)] * rfFar[i + nx * (j + ny * k)] + fvxp - fvxm + fvyp - fvym + fvzp - fvzm;

  // float rfNew = rfNear.Val(0,iw,jw,kw) + alpha*A*d;
  Real rfu2 = rfNear[i + nx * (j + ny * k)] + alpha * Au * du;
  Real rfv2 = rfFar[i + nx * (j + ny * k)] + alpha * Av * dv;

  // if(deb!=0 && i==36 && j==36 && k==36) printf("GPU %g = %g + %g %g (ot=%g)
  // %g,%g,%g,%g,%g,%g,%g,%g\n",rfu2,rfNear[i+nx*(j+nz*k)],Au,du,rfOT[i+nx*(j+nz*k)],dx*rs[i+nx*(j+nz*k)],abc[i+nx*(j+nz*k)]*rfNear[i+nx*(j+nz*k)],fuxp,fuxm,fuyp,fuym,fuzp,fuzm);

  if (lastIteration) {
    if (rfu2 > facOverOT * rfOT[i + nx * (j + ny * k)]) rfu2 = facOverOT * rfOT[i + nx * (j + ny * k)];
    if (rfv2 > 1) rfv2 = 1;
  }
  rfNearNew[i + nx * (j + ny * k)] = (rfu2 < 0 ? 0 : rfu2);
  rfFarNew[i + nx * (j + ny * k)]  = (rfv2 < 0 ? 0 : rfv2);
}



/*
  // From Altair
  template<bool Split> GPU_DEVICE_DECL inline void GLFStepPij(
      int ic, int jc, int kc,
      int nw, int nw3, float dx,
      float cdt2dxRSL, float gamma,
      const float* GPU_RESTRICT_DECL rs,
      const float* GPU_RESTRICT_DECL abc,
      const float* GPU_RESTRICT_DECL pij_,
      const float* GPU_RESTRICT_DECL qijk_,
      float* GPU_RESTRICT_DECL pijNew_,
      int deb)
*/

//
// cdt2dxRSL is cbar*dt/dx, cbar is the effective spped of light which can be less than c is the reduced speed of light approximation is sued.
// gamma is the parameter for the semi-implicit scheme:
// v_1 = v_0 + J*(gamma*v_1+(1-gamma)*v_0), J is the Jacobian
//
//  gamma=0 is the Aubert & Teyssier 2008 scheme.
//
//
//
/* split template
template<bool Split> void __global__ StepRFiIteration_Kernel( int nx, int ny, int nz, int n_ghost, 
                                                              Real cdt2dxRSL, Real gamma,
                                                              const Real* __restrict__ rs,
                                                              const Real* __restrict__ rfi,
                                                              const Real* __restrict__ abc,
                                                              const Real* __restrict__ pij_,
                                                              Real* __restrict__ rfiNew, int deb)
*/

#ifdef M1
// This is our M1 kernel, which is called
// for each of 4 frequencies
void __global__ StepRFiIteration_Kernel(int nx, int ny, int nz, int n_ghost, 
                                        Real dx, Real cdt2dxRSL, Real gamma,
                                        const Real* __restrict__ rs,
                                        const Real* __restrict__ rfi,
                                        const Real* __restrict__ abc,
                                        const Real* __restrict__ pij_,
                                        Real* __restrict__ rfiNew, int deb)
{
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int nc = nx - 2*n_ghost;
  const int jkc = tid/nc; // May need to be updated for ny and nz separately
  const int ic = n_ghost + tid%nc;
  const int jc = n_ghost + jkc%nc;
  const int kc = n_ghost + jkc/nc;

  const bool Split = true; // semi-implicit
  const int ip = ic + 1;
  const int im = ic - 1;
  const int jp = jc + 1;
  const int jm = jc - 1;
  const int kp = kc + 1;
  const int km = kc - 1;

  const int nw3 = nx*ny*nz;

  const Real* rf = rfi;
  const Real* fi[3] = { rfi+nw3, rfi+2*nw3, rfi+3*nw3 }; // fluxes
  const Real* pij[6] = { pij_, pij_+nw3, pij_+2*nw3, pij_+3*nw3, pij_+4*nw3, pij_+5*nw3 };
  Real* fiNew[3] = { rfiNew+nw3, rfiNew+2*nw3, rfiNew+3*nw3 };

  const Real d = dx*rs[ic+nx*(jc+ny*kc)] + 0.5f*(fi[0][im+nx*(jc+ny*kc)]-fi[0][ip+nx*(jc+ny*kc)]+fi[1][ic+nx*(jm+ny*kc)]-fi[1][ic+nx*(jp+ny*kc)]+fi[2][ic+nx*(jc+ny*km)]-fi[2][ic+nx*(jc+ny*kp)]) + 0.5f*(rf[im+nx*(jc+ny*kc)]+rf[ip+nx*(jc+ny*kc)]+rf[ic+nx*(jm+ny*kc)]+rf[ic+nx*(jp+ny*kc)]+rf[ic+nx*(jc+ny*km)]+rf[ic+nx*(jc+ny*kp)]-6*rf[ic+nx*(jc+ny*kc)]);

  float rf1, cdt2dx, w1, w2;
  if(Split)
  {
      cdt2dx = cdt2dxRSL/(1+cdt2dxRSL*gamma*3);
      Real tau = cdt2dx*abc[ic+nx*(jc+ny*kc)];
      w1 = expf(-tau);
      w2 = (tau<0.1f ? 1-0.5f*tau*(1-(1.0f/3.0f)*tau*(1-0.25f*tau)) : (1-w1)/tau);

      rf1 = rf[ic+nx*(jc+ny*kc)]*w1 + cdt2dx*d*w2;
  }
  else
  {
      cdt2dx = cdt2dxRSL/(1+cdt2dxRSL*gamma*(abc[ic+nx*(jc+ny*kc)]+3));

      rf1 = rf[ic+nx*(jc+ny*kc)] + cdt2dx*(d-abc[ic+nx*(jc+ny*kc)]*rf[ic+nx*(jc+ny*kc)]);
  }


  rfiNew[ic+nx*(jc+ny*kc)] = (rf1<0 ? 0 : rf1);

  for(int m=0; m<3; m++)
  {
      constexpr int ix[] = { 0, 1, 3 };
      constexpr int iy[] = { 1, 2, 4 };
      constexpr int iz[] = { 3, 4, 5 };

      const Real df = 0.5f*(pij[ix[m]][im+nx*(jc+ny*kc)]-pij[ix[m]][ip+nx*(jc+ny*kc)]+pij[iy[m]][ic+nx*(jm+ny*kc)]-pij[iy[m]][ic+nx*(jp+ny*kc)]+pij[iz[m]][ic+nx*(jc+ny*km)]-pij[iz[m]][ic+nx*(jc+ny*kp)]) + 0.5f*(fi[m][im+nx*(jc+ny*kc)]+fi[m][ip+nx*(jc+ny*kc)]+fi[m][ic+nx*(jm+ny*kc)]+fi[m][ic+nx*(jp+ny*kc)]+fi[m][ic+nx*(jc+ny*km)]+fi[m][ic+nx*(jc+ny*kp)]-6*fi[m][ic+nx*(jc+ny*kc)]);

      if(Split)
      {
          fiNew[m][ic+nx*(jc+ny*kc)] = fi[m][ic+nx*(jc+ny*kc)]*w1 + cdt2dx*df*w2;
      }
      else
      {
          fiNew[m][ic+nx*(jc+ny*kc)] = fi[m][ic+nx*(jc+ny*kc)] + cdt2dx*(df-abc[ic+nx*(jc+ny*kc)]*fi[m][ic+nx*(jc+ny*kc)]);
      }
  }
}


// pij kernel implementation

/* after pij, limit to total flux
                    //
                    //  Final data copy
                    //
                    const int nout3 = nout*nout*nout;
                    const int orig = (nw-nout)/2; // not offset, this is used for windows of different size
                    const int nmax = orig + nout;
                    if(ic>=orig && jc>=orig && kc>=orig && ic<nmax && jc<nmax && kc<nmax)
                    {
                        const int idx = (ic-orig) + nout*(jc-orig+nout*(kc-orig));

                        rfiOut[idx] = rfi[ic+nw*(jc+nw*kc)];
                        for(int m=0; m<3; m++)
                        {
                            rfiOut[idx+(m+1)*nout3] = fminf(fmaxf(rfi[ic+nw*(jc+nw*kc)+nw3*(m+1)],-rfi[ic+nw*(jc+nw*kc)]),rfi[ic+nw*(jc+nw*kc)]);
                        }
                    }
                }
*/

/* after pij, limit to total flux
//
//  Final data copy
//
const int nout3 = nout*nout*nout;
const int orig = (nw-nout)/2; // not offset, this is used for windows of different size
const int nmax = orig + nout;
if(ic>=orig && jc>=orig && kc>=orig && ic<nmax && jc<nmax && kc<nmax)
{
    const int idx = (ic-orig) + nout*(jc-orig+nout*(kc-orig));

    rfiOut[idx] = rfi[ic+nw*(jc+nw*kc)];
    for(int m=0; m<3; m++)
    {
        rfiOut[idx+(m+1)*nout3] = fminf(fmaxf(rfi[ic+nw*(jc+nw*kc)+nw3*(m+1)],
                                             -rfi[ic+nw*(jc+nw*kc)]),
                                              rfi[ic+nw*(jc+nw*kc)]);
    }
}
}
*/

//
//  This is called ClipRFi in Altair
//
void __global__ ClipRFi_Kernel(int nx, int ny, int nz, int n_ghost, const Real* __restrict__ rfi, int nout, Real* __restrict__ rfiOut, int deb)
{
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int nc = nx - 2*n_ghost;
  const int jkc = tid/nc; // May need to be updated for ny and nz separately
  const int ic = n_ghost + tid%nc;
  const int jc = n_ghost + jkc%nc;
  const int kc = n_ghost + jkc/nc;
  if(kc >= nz-n_ghost) return;
  const int nout3 = n_ghost*n_ghost*n_ghost;
  const int origx = (nz-n_ghost)/2; // not offset, this is used for windows of different size
  const int origy = (ny-n_ghost)/2; // not offset, this is used for windows of different size
  const int origz = (nz-n_ghost)/2; // not offset, this is used for windows of different size

  const int nmaxx = origx + n_ghost;
  const int nmaxy = origy + n_ghost;
  const int nmaxz = origz + n_ghost;

  const int nw3 = nx*ny*nz;

  if(ic>=origx && jc>=origy && kc>=origx && ic<nmaxx && jc<nmaxy && kc<nmaxz)
  {
      const int idx = (ic-origx) + nout*(jc-origy+nout*(kc-origz));

      rfiOut[idx] = rfi[ic+nx*(jc+ny*kc)];
      for(int m=0; m<3; m++)
      {
          rfiOut[idx+(m+1)*nout3] = fminf(fmaxf(rfi[ic+nx*(jc+ny*kc)+nw3*(m+1)],
                                               -rfi[ic+nx*(jc+ny*kc)]),
                                                rfi[ic+nx*(jc+ny*kc)]);
      }
  }
}


/* From Altair

//
//  Compute pressure tensor - has to be a separate kernel since
pij is needed in its entirety for the step
//
template<class PijFunctor> GPU_KERNEL_DECL void GLFMakeP(
    int offset, int nw, int nw3, float dx,
    const float* GPU_RESTRICT_DECL rfi,
    float* GPU_RESTRICT_DECL pij,
    PijFunctor pf,
    int deb)
{
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    const int nc = nw - 2*offset;
    const int jkc = tid/nc;
    const int ic = offset + tid%nc;
    const int jc = offset + jkc%nc;
    const int kc = offset + jkc/nc;
    if(kc >= nw-offset) return;

    pf(offset,nw,nw3,ic,jc,kc,rfi,pij,deb);
}
};

*/


//  Compute pressure tensor - has to be a separate kernel since
//  pij is needed in its entirety for the step
template<class PijFunctor> void __global__ GLFMakeP_Kernel(int nx, int ny, int nz, int n_ghost, float dx,
                                                    const float* rfi, float* pij, PijFunctor pf, int deb)
{
    const int nw3 = nx*ny*nz;
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    const int nc = nx - 2*n_ghost;
    const int jkc = tid/nc;
    const int ic = n_ghost + tid%nc;
    const int jc = n_ghost + jkc%nc;
    const int kc = n_ghost + jkc/nc;
    if(kc >= nx-n_ghost) return;

    pf(n_ghost,nx,ny,nz,ic,jc,kc,rfi,pij,deb);
}


/*
struct DEVICE_ALIGN_DECL PijFunctorM1
{
    __global__ void operator()(int offset, int nx, int ny, int nz, 
                    int ic, int jc, int kc, const Real* rfi, Real* pij, int deb)
    {
        if(ic<offset || jc<offset || kc<offset || ic>=nx-offset || jc>=ny-offset || kc>=nz-offset) return;

        const int nw3 = nx*ny*nz;
        const int idx = ic + nx*(jc+ny*kc);
        const float r =  rfi[idx];
        const float fx = rfi[idx+1*nw3];
        const float fy = rfi[idx+2*nw3];
        const float fz = rfi[idx+3*nw3];

        if(r > 0)
        {
            float flux2 = fx*fx + fy*fy + fz*fz;
            if(flux2 > 0)
            {
                float f2 = min(flux2/(r*r),1.0f);

                float alpha = (3+4*f2)/(5+2*sqrt(4-3*f2));
                float wd = (1-alpha)/2*r;
                float wn = (3*alpha-1)/2*r/flux2;

                pij[idx+0*nw3] = wn*fx*fx + wd;
                pij[idx+1*nw3] = wn*fy*fx;
                pij[idx+2*nw3] = wn*fy*fy + wd;
                pij[idx+3*nw3] = wn*fz*fx;
                pij[idx+4*nw3] = wn*fz*fy;
                pij[idx+5*nw3] = wn*fz*fz + wd;
            }
            else
            {
                pij[idx+0*nw3] = r/3;
                pij[idx+1*nw3] = 0;
                pij[idx+2*nw3] = r/3;
                pij[idx+3*nw3] = 0;
                pij[idx+4*nw3] = 0;
                pij[idx+5*nw3] = r/3;
            }
        }
        else
        {
            pij[idx+0*nw3] = 0;
            pij[idx+1*nw3] = 0;
            pij[idx+2*nw3] = 0;
            pij[idx+3*nw3] = 0;
            pij[idx+4*nw3] = 0;
            pij[idx+5*nw3] = 0;
        }
    }
};
*/
#endif //M1


#endif  // RT
