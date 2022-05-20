#pragma once

#include "HenryPeriodic.hpp"

class ParisPeriodic {
  public:
    ParisPeriodic(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
    size_t bytes() const { return henry.bytes(); }
    void solve(size_t bytes, double *density, double *potential) const;
  private:
    int ni_,nj_;
#if defined(PARIS_3PT) || defined(PARIS_5PT)
    int nk_;
#endif
    double ddi_,ddj_,ddk_;
    HenryPeriodic henry;
};
