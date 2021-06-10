#pragma once

#include <mpi.h>
#include "gpu.hpp"

class PoissonPeriodic3DBlockedGPU {
  public:
    PoissonPeriodic3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
    ~PoissonPeriodic3DBlockedGPU();
    void solve(long bytes, double *da, double *db);
    long bytes() const { return bytes_; }
  private:
    MPI_Comm commSlab_,commWorld_;
    double di_,dj_,dk_;
    int mi_,mj_,mk_;
    int ni_,nj_,nk_;
    long bytes_;
    cufftHandle dz2d_,zd2d_,zz1d_;
#ifndef MPI_GPU
    double *ha_, *hb_;
#endif
};

