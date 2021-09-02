#pragma once

#include <mpi.h>

#include "../../utils/gpu.hpp"

class PoissonPeriodic3x1DBlockedGPU {
  public:
    PoissonPeriodic3x1DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
    ~PoissonPeriodic3x1DBlockedGPU();
    size_t bytes() const { return bytes_; }
    void solve(size_t bytes, double *density, double *potential) const;
  private:
    double ddi_,ddj_,ddk_;
    int idi_,idj_,idk_;
    int mi_,mj_,mk_;
    int nh_,ni_,nj_,nk_;
    int mp_,mq_;
    int idp_,idq_;
    MPI_Comm commI_,commJ_,commK_;
    int di_,dj_,dk_;
    int dhq_,dip_,djp_,djq_;
    size_t bytes_;
    cufftHandle c2ci_,c2cj_,c2rk_,r2ck_;
#ifndef MPI_GPU
    double *ha_, *hb_;
#endif
};
