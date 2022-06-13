#ifdef PARIS


#include "fft_3D.h"
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include <cassert>
#include <cfloat>
#include <climits>

FFT_3D::FFT_3D():
  dn_{0,0,0},
  dr_{0,0,0},
  lo_{0,0,0},
  lr_{0,0,0},
  myLo_{0,0,0},
  henry_(nullptr),
  minBytes_(0),
  inputBytes_(0),
  outputBytes_(0),
  da_(nullptr),
  db_(nullptr)
{}

  
void FFT_3D::Initialize(const Real lx, const Real ly, const Real lz, const Real xMin, const Real yMin, const Real zMin, const int nx, const int ny, const int nz, const int nxReal, const int nyReal, const int nzReal, const Real dx, const Real dy, const Real dz)
{
  
  const long nl012 = long(nxReal)*long(nyReal)*long(nzReal);
  assert(nl012 <= INT_MAX);

  dn_[0] = nzReal;
  dn_[1] = nyReal;
  dn_[2] = nxReal;

  dr_[0] = dz;
  dr_[1] = dy;
  dr_[2] = dx;

  lr_[0] = lz;
  lr_[1] = ly;
  lr_[2] = lx;

  myLo_[0] = zMin;
  myLo_[1] = yMin;
  myLo_[2] = xMin;
  MPI_Allreduce(myLo_,lo_,3,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);

  const Real hi[3] = {lo_[0]+lz-dr_[0],lo_[1]+ly-dr_[1],lo_[2]+lx-dr_[2]};
  const int n[3] = {nz,ny,nx};
  const int m[3] = {n[0]/nzReal,n[1]/nyReal,n[2]/nxReal};
  const int id[3] = {int(round((zMin-lo_[0])/(dn_[0]*dr_[0]))),int(round((yMin-lo_[1])/(dn_[1]*dr_[1]))),int(round((xMin-lo_[2])/(dn_[2]*dr_[2])))};
  chprintf("  FFT: [ %g %g %g ]-[ %g %g %g ] N_local[ %d %d %d ] Tasks[ %d %d %d ]\n",lo_[2],lo_[1],lo_[0],lo_[2]+lx,lo_[1]+ly,lo_[0]+lz,dn_[2],dn_[1],dn_[0],m[2],m[1],m[0]);

  assert(dn_[0] == n[0]/m[0]);
  assert(dn_[1] == n[1]/m[1]);
  assert(dn_[2] == n[2]/m[2]);
  
  ni_ = n[0];
  nj_ = n[1];
  nk_ = n[2];
  
  ddi_ = 2.0*M_PI*double(n[0]-1)/(double(n[0])*(hi[0]-lo_[0]));
  ddj_ = 2.0*M_PI*double(n[1]-1)/(double(n[1])*(hi[1]-lo_[1]));
  ddk_ = 2.0*M_PI*double(n[2]-1)/(double(n[2])*(hi[2]-lo_[2]));
  
  henry_ = new HenryPeriodic(n,lo_,hi,m,id);
  assert(henry_);
  // pp_ = new ParisPeriodic(n,lo_,hi,m,id);
  // assert(pp_);
  minBytes_ = henry_->bytes();
  inputBytes_ = long(sizeof(Real))*dn_[0]*dn_[1]*dn_[2];
  // const long gg = N_GHOST_POTENTIAL+N_GHOST_POTENTIAL;
  const long gg = 0;
  outputBytes_ = long(sizeof(Real))*(dn_[0]+gg)*(dn_[1]+gg)*(dn_[2]+gg);
  
  CHECK(cudaMalloc(reinterpret_cast<void **>(&da_),std::max(minBytes_,inputBytes_)));
  assert(da_);
  
  CHECK(cudaMalloc(reinterpret_cast<void **>(&db_),std::max(minBytes_,outputBytes_)));
  assert(db_);
}  

void FFT_3D::Reset()
{
  if (db_) CHECK(cudaFree(db_));
  db_ = nullptr;

  if (da_) CHECK(cudaFree(da_));
  da_ = nullptr;

  outputBytes_ = inputBytes_ = minBytes_ = 0;

  if (henry_) delete henry_;
  henry_ = nullptr;

  myLo_[2] = myLo_[1] = myLo_[0] = 0;
  lr_[2] = lr_[1] = lr_[0] = 0;
  lo_[2] = lo_[1] = lo_[0] = 0;
  dr_[2] = dr_[1] = dr_[0] = 0;
  dn_[2] = dn_[1] = dn_[0] = 0;
}

FFT_3D::~FFT_3D() { Reset(); }





#endif

