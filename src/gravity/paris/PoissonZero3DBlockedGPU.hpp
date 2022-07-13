#pragma once

#include <mpi.h>
#include "../../utils/gpu.hpp"

class PoissonZero3DBlockedGPU {
  public:
    PoissonZero3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
    ~PoissonZero3DBlockedGPU();
    long bytes() const { return bytes_; }
    void solve(long bytes, double *density, double *potential) const;
  private:
    double ddi_,ddj_,ddk_;
    int idi_,idj_,idk_;
    int mi_,mj_,mk_;
    int ni_,nj_,nk_;
    int mp_,mq_;
    int idp_,idq_;
    MPI_Comm commI_,commJ_,commK_;
    int di_,dj_,dk_;
    int dip_,djp_,djq_,dkq_;
    int ni2_,nj2_,nk2_;
    long bytes_;
    cufftHandle d2zi_,d2zj_,d2zk_;
#ifndef MPI_GPU
    double *ha_, *hb_;
#endif
};
