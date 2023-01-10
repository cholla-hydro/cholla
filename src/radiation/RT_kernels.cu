/*LICENSE*/

#include <cstdio>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "radiation.h"
#include "RT_functions.h"


void __global__ Set_RT_Boundaries_Periodic_Kernel(int direction, int side, int n_i, int n_j, int nx, int ny, int nz, int n_ghost, int n_freq, struct Rad3D::RT_Fields rtFields)
{

  int n_cells = nx*ny*nz;
  
  // get a global thread ID
  int tid, tid_i, tid_j, tid_k, tid_src, tid_dst;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_k = tid / (n_i*n_j);
  tid_j = (tid - tid_k*n_i*n_j) / n_i;
  tid_i = tid - tid_k*n_i*n_j - tid_j*n_i;
  
  if ( tid_i < 0 || tid_i >= n_i || tid_j < 0 || tid_j >= n_j || tid_k < 0 || tid_k >= n_ghost ) return;
  
  if ( direction == 0 ){
    if ( side == 0 ) tid_src = ( nx - 2*n_ghost + tid_k )  + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = ( tid_k )                   + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = ( n_ghost + tid_k  )        + (tid_i)*nx  + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = ( nx - n_ghost + tid_k )    + (tid_i)*nx  + (tid_j)*nx*ny;
  }
  if ( direction == 1 ){
    if ( side == 0 ) tid_src = (tid_i) + ( ny - 2*n_ghost + tid_k  )*nx  + (tid_j)*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + ( tid_k )*nx                    + (tid_j)*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + ( n_ghost + tid_k  )*nx         + (tid_j)*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + ( ny - n_ghost + tid_k )*nx     + (tid_j)*nx*ny;
  }
  if ( direction == 2 ){
    if ( side == 0 ) tid_src = (tid_i) + (tid_j)*nx + ( nz - 2*n_ghost + tid_k  )*nx*ny;
    if ( side == 0 ) tid_dst = (tid_i) + (tid_j)*nx + ( tid_k  )*nx*ny;
    if ( side == 1 ) tid_src = (tid_i) + (tid_j)*nx + ( n_ghost + tid_k  )*nx*ny;
    if ( side == 1 ) tid_dst = (tid_i) + (tid_j)*nx + ( nz - n_ghost + tid_k  )*nx*ny;
  }
  
  for (int i=0; i<n_freq; i++) {
    rtFields.dev_rf[tid_dst+(1+i)*n_cells] = rtFields.dev_rf[tid_src+(1+i)*n_cells];
    rtFields.dev_rf[tid_dst+(1+n_freq+i)*n_cells] = rtFields.dev_rf[tid_src+(1+n_freq+i)*n_cells];
  }
}


void __global__ Calc_Absorption_Kernel(int nx, int ny, int nz, Real dx, CrossSectionInCU xs, const Real* __restrict__ dens, Real* __restrict__ abc)
{
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    const int jk = tid/nx;
    const int i = tid%nx;
    const int j = jk%ny;
    const int k = jk/ny;
    
    if(k >= nz) return;
    
    const Real densHI = dens[i+nx*(j+ny*(k+0*nz))];
    const Real densHeI = dens[i+nx*(j+ny*(k+2*nz))];
    const Real densHeII = dens[i+nx*(j+ny*(k+3*nz))];

    abc[i+nx*(j+ny*(k+0*nz))] = dx*(xs.HIatHI*densHI);
    abc[i+nx*(j+ny*(k+1*nz))] = dx*(xs.HIatHeI*densHI+xs.HeIatHeI*densHeI);
    abc[i+nx*(j+ny*(k+2*nz))] = dx*(xs.HIatHeII*densHI+xs.HeIatHeII*densHeI+xs.HeIIatHeII*densHeII);
}


#define PU(TT,I,J,K)  rfNear[i+I+nx*(j+J+nz*(k+K))]*et##TT[i+I+nx*(j+J+nz*(k+K))]
#define PV(TT,I,J,K)  rfFar[i+I+nx*(j+J+nz*(k+K))]*(1.0f/3.0f) // et for the far field is unitary matrix/3


void __global__ OTVETIteration_Kernel(int nx, int ny, int nz, int n_ghost,
    Real dx,
    bool lastIteration,
    const Real rsFarFactor,
    const Real* __restrict__ rs,
    const Real* __restrict__ et,
    const Real* __restrict__ rfOT,
    const Real* __restrict__ rfNear,
    const Real* __restrict__ rfFar,
    const Real* __restrict__ abc,
    Real* __restrict__ rfNearNew,
    Real* __restrict__ rfFarNew,
    int deb)
{
    const Real alpha = 0.8; // Parameters from cpp code
    const Real gamma = 1;
    const Real epsNum = 1.0e-6;
    const Real facOverOT = 1.5;
    
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    const int jk = tid/nx;
    const int i = tid%nx;
    const int j = jk%ny;
    const int k = jk/ny;
    
    if(i<n_ghost || j<n_ghost || k<n_ghost || i>=nx-n_ghost || j>=ny-n_ghost || k>=nz-n_ghost) return;
    
    const int fieldPitch = nx*ny*nz;
    
    //
    //  Set pointers into et array pointing to specific fields.
    //  Names are the same as in cpp code
    //
    const Real *etXX = et + 0*fieldPitch;
    const Real *etXY = et + 1*fieldPitch;
    const Real *etYY = et + 2*fieldPitch;
    const Real *etXZ = et + 3*fieldPitch;
    const Real *etYZ = et + 4*fieldPitch;
    const Real *etZZ = et + 5*fieldPitch;
    
    //
    //  Compute edge projections U^(x,y,z)_{i,j,k}
    //
    
    //
    //  X-direction
    //
    /// float ahx = epsNum + 0.5f*(abc.Val(0,i,j,k)+abc.Val(0,i+1,j,k));
    const Real ahpcc = epsNum + 0.5f*(abc[i+nx*(j+nz*k)]+abc[i+1+nx*(j+nz*k)]);
    const Real ahmcc = epsNum + 0.5f*(abc[i+nx*(j+nz*k)]+abc[i-1+nx*(j+nz*k)]);
    
    /// float ux = rfNear.Val(0,i+1,j,k)*et.Val(0,i+1,j,k) - rfNear.Val(0,i,j,k)*et.Val(0,i,j,k);
    Real uxp = PU(XX, 1, 0, 0) - PU(XX, 0, 0, 0);
    Real uxm = PU(XX, 0, 0, 0) - PU(XX,-1, 0, 0);
    Real vxp = PV(XX, 1, 0, 0) - PV(XX, 0, 0, 0);
    Real vxm = PV(XX, 0, 0, 0) - PV(XX,-1, 0, 0);
    
    /// float uy = 0.25f*(rfNear.Val(0,i+1,j+1,k)*et.Val(1,i+1,j+1,k) + rfNear.Val(0,i,j+1,k)*et.Val(1,i,j+1,k) -
    ///                   rfNear.Val(0,i+1,j-1,k)*et.Val(1,i+1,j-1,k) - rfNear.Val(0,i,j-1,k)*et.Val(1,i,j-1,k));
    Real uyp = 0.25f*(PU(XY, 1, 1, 0)+PU(XY, 0, 1, 0)-PU(XY, 1,-1, 0)-PU(XY, 0,-1, 0));
    Real uym = 0.25f*(PU(XY, 0, 1, 0)+PU(XY,-1, 1, 0)-PU(XY, 0,-1, 0)-PU(XY,-1,-1, 0));
    Real vyp = 0;
    Real vym = 0;
    
    /// float uz = 0.25f*(rfNear.Val(0,i+1,j,k+1)*et.Val(3,i+1,j,k+1) + rfNear.Val(0,i,j,k+1)*et.Val(3,i,j,k+1) -
    ///                   rfNear.Val(0,i+1,j,k-1)*et.Val(3,i+1,j,k-1) - rfNear.Val(0,i,j,k-1)*et.Val(3,i,j,k-1));
    Real uzp = 0.25f*(PU(XZ, 1, 0, 1)+PU(XZ, 0, 0, 1)-PU(XZ, 1, 0,-1)-PU(XZ, 0, 0,-1));
    Real uzm = 0.25f*(PU(XZ, 0, 0, 1)+PU(XZ,-1, 0, 1)-PU(XZ, 0, 0,-1)-PU(XZ,-1, 0,-1));
    Real vzp = 0;
    Real vzm = 0;
    
    /// flux.Val(0,i,j,k) = (ux+uy+uz)/ahx;  
    const Real fuxp = (uxp+uyp+uzp)/ahpcc;
    const Real fuxm = (uxm+uym+uzm)/ahmcc;
    const Real fvxp = (vxp+vyp+vzp)/ahpcc;
    const Real fvxm = (vxm+vym+vzm)/ahmcc;
    
    //
    //  Y-direction
    //
    /// float ahy = epsNum + 0.5f*(abc.Val(0,i,j,k)+abc.Val(0,i,j+1,k));
    const Real ahcpc = epsNum + 0.5f*(abc[i+nx*(j+nz*k)]+abc[i+nx*(j+1+nz*k)]);
    const Real ahcmc = epsNum + 0.5f*(abc[i+nx*(j+nz*k)]+abc[i+nx*(j-1+nz*k)]);
    
    /// float uy = rfNear.Val(0,i,j+1,k)*et.Val(2,i,j+1,k) - rfNear.Val(0,i,j,k)*et.Val(2,i,j,k);
    uyp = PU(YY, 0, 1, 0) - PU(YY, 0, 0, 0);
    uym = PU(YY, 0, 0, 0) - PU(YY, 0,-1, 0);
    vyp = PV(YY, 0, 1, 0) - PV(YY, 0, 0, 0);
    vym = PV(YY, 0, 0, 0) - PV(YY, 0,-1, 0);
    
    /// float ux = 0.25f*(rfNear.Val(0,i+1,j+1,k)*et.Val(1,i+1,j+1,k) + rfNear.Val(0,i+1,j,k)*et.Val(1,i+1,j,k) -
    ///                   rfNear.Val(0,i-1,j+1,k)*et.Val(1,i-1,j+1,k) - rfNear.Val(0,i-1,j,k)*et.Val(1,i-1,j,k));
    uxp = 0.25f*(PU(XY, 1, 1, 0)+PU(XY, 1, 0, 0)-PU(XY,-1, 1, 0)-PU(XY,-1, 0, 0));
    uxm = 0.25f*(PU(XY, 1, 0, 0)+PU(XY, 1,-1, 0)-PU(XY,-1, 0, 0)-PU(XY,-1,-1, 0));
    vxp = 0;
    vxm = 0;
    
    /// float uz = 0.25f*(rfNear.Val(0,i,j+1,k+1)*et.Val(4,i,j+1,k+1) + rfNear.Val(0,i,j,k+1)*et.Val(4,i,j,k+1) -
    ///                   rfNear.Val(0,i,j+1,k-1)*et.Val(4,i,j+1,k-1) - rfNear.Val(0,i,j,k-1)*et.Val(4,i,j,k-1));
    uzp = 0.25f*(PU(YZ, 0, 1, 1)+PU(YZ, 0, 0, 1)-PU(YZ, 0, 1,-1)-PU(YZ, 0, 0,-1));
    uzm = 0.25f*(PU(YZ, 0, 0, 1)+PU(YZ, 0,-1, 1)-PU(YZ, 0, 0,-1)-PU(YZ, 0,-1,-1));
    vzp = 0;
    vzm = 0;
    
    /// flux.Val(1,i,j,k) = (ux+uy+uz)/ahy;
    const Real fuyp = (uxp+uyp+uzp)/ahcpc;
    const Real fuym = (uxm+uym+uzm)/ahcmc;
    const Real fvyp = (vxp+vyp+vzp)/ahcpc;
    const Real fvym = (vxm+vym+vzm)/ahcmc;
    
    //
    //  Z-direction
    //
    /// float ahz = epsNum + 0.5f*(abc.Val(0,i,j,k)+abc.Val(0,i,j,k+1));
    const Real ahccp = epsNum + 0.5f*(abc[i+nx*(j+nz*k)]+abc[i+nx*(j+nz*(k+1))]);
    const Real ahccm = epsNum + 0.5f*(abc[i+nx*(j+nz*k)]+abc[i+nx*(j+nz*(k-1))]);
    
    /// float uz = rfNear.Val(0,i,j,k+1)*et.Val(5,i,j,k+1) - rfNear.Val(0,i,j,k)*et.Val(5,i,j,k);
    uzp = PU(ZZ, 0, 0, 1) - PU(ZZ, 0, 0, 0);
    uzm = PU(ZZ, 0, 0, 0) - PU(ZZ, 0, 0,-1);
    vzp = PV(ZZ, 0, 0, 1) - PV(ZZ, 0, 0, 0);
    vzm = PV(ZZ, 0, 0, 0) - PV(ZZ, 0, 0,-1);
    
    /// float ux = 0.25f*(rfNear.Val(0,i+1,j,k+1)*et.Val(3,i+1,j,k+1) + rfNear.Val(0,i+1,j,k)*et.Val(3,i+1,j,k) -
    ///                   rfNear.Val(0,i-1,j,k+1)*et.Val(3,i-1,j,k+1) - rfNear.Val(0,i-1,j,k)*et.Val(3,i-1,j,k));
    uxp = 0.25f*(PU(XZ, 1, 0, 1)+PU(XZ, 1, 0, 0)-PU(XZ,-1, 0, 1)-PU(XZ,-1, 0, 0));
    uxm = 0.25f*(PU(XZ, 1, 0, 0)+PU(XZ, 1, 0,-1)-PU(XZ,-1, 0, 0)-PU(XZ,-1, 0,-1));
    vxp = 0;
    vxm = 0;
    
    /// float uy = 0.25f*(rfNear.Val(0,i,j+1,k+1)*et.Val(4,i,j+1,k+1) + rfNear.Val(0,i,j+1,k)*et.Val(4,i,j+1,k) -
    ///                   rfNear.Val(0,i,j-1,k+1)*et.Val(4,i,j-1,k+1) - rfNear.Val(0,i,j-1,k)*et.Val(4,i,j-1,k));
    uyp = 0.25f*(PU(YZ, 0, 1, 1)+PU(YZ, 0, 1, 0)-PU(YZ, 0,-1, 1)-PU(YZ, 0,-1, 0));
    uym = 0.25f*(PU(YZ, 0, 1, 0)+PU(YZ, 0, 1,-1)-PU(YZ, 0,-1, 0)-PU(YZ, 0,-1,-1));
    vyp = 0;
    vym = 0;
    
    /// flux.Val(2,i,j,k) = (ux+uy+uz)/ahz;
    const Real fuzp = (uxp+uyp+uzp)/ahccp;
    const Real fuzm = (uxm+uym+uzm)/ahccm;
    const Real fvzp = (vxp+vyp+vzp)/ahccp;
    const Real fvzm = (vxm+vym+vzm)/ahccm;
    
    /// float minus_wII = et.Val(0,iw,jw,kw)*(abch.Val(0,ihm,jw,kw)+abch.Val(0,ihp,jw,kw)) + et.Val(2,iw,jw,kw)*(abch.Val(1,iw,jhm,kw)+abch.Val (1,iw,jhp,kw)) + et.Val(5,iw,jw,kw)*(abch.Val(2,iw,jw,khm)+abch.Val(2,iw,jw,khp));
    const Real uminus_wII = etXX[i+nx*(j+nz*k)]*(1/ahpcc+1/ahmcc) + etYY[i+nx*(j+nz*k)]*(1/ahcpc+1/ahcmc) + etZZ[i+nx*(j+nz*k)]*(1/ahccp+1/ahccm);
    const Real vminus_wII = (1.0f/3.0f)/ahpcc + (1.0f/3.0f)/ahmcc + (1.0f/3.0f)/ahcpc + (1.0f/3.0f)/ahcmc + (1.0f/3.0f)/ahccp + (1.0f/3.0f)/ahccm;

    /// float A = gamma/(1+gamma*(abc.Val(0,iw,jw,kw)+minus_wII));
    const Real Au = gamma/(1+gamma*(abc[i+nx*(j+nz*k)]+uminus_wII));
    const Real Av = gamma/(1+gamma*(abc[i+nx*(j+nz*k)]+vminus_wII));
    
    /// float d = pars.dx*rs.Val(0,iw,jw,kw) - abc.Val(0,iw,jw,kw)*rfNear.Val(0,iw,jw,kw) + flux.Val(0,ihp,jw,kw) - flux.Val(0,ihm,jw,kw) + flux.Val(1,iw,jhp,kw) - flux.Val(1,iw,jhm,kw) + flux.Val(2,iw,jw,khp) - flux.Val(2,iw,jw,khm);
    Real du = dx*rs[i+nx*(j+nz*k)] - abc[i+nx*(j+nz*k)]*rfNear[i+nx*(j+nz*k)] + fuxp - fuxm + fuyp - fuym + fuzp - fuzm;
    Real dv = dx*rsFarFactor*rfFar[i+nx*(j+nz*k)] - abc[i+nx*(j+nz*k)]*rfFar[i+nx*(j+nz*k)] + fvxp - fvxm + fvyp - fvym + fvzp - fvzm;

    /// float rfNew = rfNear.Val(0,iw,jw,kw) + alpha*A*d;
    Real rfu2 = rfNear[i+nx*(j+nz*k)] + alpha*Au*du;
    Real rfv2 = rfFar[i+nx*(j+nz*k)] + alpha*Av*dv;
    
    ///if(deb!=0 && i==37 && j==36 && k==36) printf("GPU %g = %g + %g %g (ot=%g) %g,%g,%g,%g,%g,%g,%g,%g\n",rfu2,rfNear[i+nx*(j+nz*k)],Au,du,rfOT[i+nx*(j+nz*k)],dx*rs[i+nx*(j+nz*k)],abc[i+nx*(j+nz*k)]*rfNear[i+nx*(j+nz*k)],fuxp,fuxm,fuyp,fuym,fuzp,fuzm);

    if(lastIteration)
    {
        if(rfu2 > facOverOT*rfOT[i+nx*(j+nz*k)]) rfu2 = facOverOT*rfOT[i+nx*(j+nz*k)];
        if(rfv2 > 1) rfv2 = 1;
    }
    rfNearNew[i+nx*(j+ny*k)] = (rfu2<0 ? 0 : rfu2);
    rfFarNew[i+nx*(j+ny*k)]  = (rfv2<0 ? 0 : rfv2);
}
