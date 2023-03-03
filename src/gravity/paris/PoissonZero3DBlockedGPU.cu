#ifdef PARIS_GALACTIC

  #include <algorithm>
  #include <cassert>
  #include <cmath>
  #include <cstdio>
  #include <cstdlib>

  #include "PoissonZero3DBlockedGPU.hpp"

static constexpr double sqrt2 = 0.4142135623730950488016887242096980785696718753769480731766797379;

static inline __host__ __device__ double sqr(const double x) { return x * x; }

PoissonZero3DBlockedGPU::PoissonZero3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3],
                                                 const int id[3])
    :
  #ifdef PARIS_GALACTIC_3PT
      ddi_(2.0 * double(n[0] - 1) / (hi[0] - lo[0])),
      ddj_(2.0 * double(n[1] - 1) / (hi[1] - lo[1])),
      ddk_(2.0 * double(n[2] - 1) / (hi[2] - lo[2])),
  #elif defined PARIS_GALACTIC_5PT
      ddi_(sqr(double(n[0] - 1) / (hi[0] - lo[0])) / 6.0),
      ddj_(sqr(double(n[1] - 1) / (hi[1] - lo[1])) / 6.0),
      ddk_(sqr(double(n[2] - 1) / (hi[2] - lo[2])) / 6.0),
  #else
      ddi_{M_PI * double(n[0] - 1) / (double(n[0]) * (hi[0] - lo[0]))},
      ddj_{M_PI * double(n[1] - 1) / (double(n[1]) * (hi[1] - lo[1]))},
      ddk_{M_PI * double(n[2] - 1) / (double(n[2]) * (hi[2] - lo[2]))},
  #endif
      idi_(id[0]),
      idj_(id[1]),
      idk_(id[2]),
      mi_(m[0]),
      mj_(m[1]),
      mk_(m[2]),
      ni_(n[0]),
      nj_(n[1]),
      nk_(n[2])
{
  mq_ = int(round(sqrt(mk_)));
  while (mk_ % mq_) {
    mq_--;
  }
  mp_ = mk_ / mq_;
  assert(mp_ * mq_ == mk_);

  idp_ = idk_ / mq_;
  idq_ = idk_ % mq_;

  {
    const int color = idi_ * mj_ + idj_;
    const int key   = idk_;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &commK_);
  }
  {
    const int color = idi_ * mp_ + idp_;
    const int key   = idj_ * mq_ + idq_;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &commJ_);
  }
  {
    const int color = idj_ * mq_ + idq_;
    const int key   = idi_ * mp_ + idp_;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &commI_);
  }
  di_ = (ni_ + mi_ - 1) / mi_;
  dj_ = (nj_ + mj_ - 1) / mj_;
  dk_ = (nk_ + mk_ - 1) / mk_;

  dip_          = (di_ + mp_ - 1) / mp_;
  djq_          = (dj_ + mq_ - 1) / mq_;
  const int mjq = mj_ * mq_;
  dkq_          = (nk_ + mjq - 1) / mjq;
  const int mip = mi_ * mp_;
  djp_          = (nj_ + mip - 1) / mip;

  ni2_ = 2 * (ni_ / 2 + 1);
  nj2_ = 2 * (nj_ / 2 + 1);
  nk2_ = 2 * (nk_ / 2 + 1);

  const long nMax = std::max({di_ * dj_ * dk_, dip_ * djq_ * mk_ * dk_, dip_ * mp_ * djq_ * mq_ * dk_,
                              dip_ * djq_ * nk2_, dip_ * djq_ * mjq * dkq_, dip_ * dkq_ * nj2_,
                              dip_ * dkq_ * mip * djp_, dkq_ * djp_ * mip * dip_, dkq_ * djp_ * ni2_});
  bytes_          = nMax * sizeof(double);

  int nkh = nk_ / 2 + 1;
  CHECK(cufftPlanMany(&d2zk_, 1, &nk_, &nk_, 1, nk_, &nkh, 1, nkh, CUFFT_D2Z, dip_ * djq_));
  int njh = nj_ / 2 + 1;
  CHECK(cufftPlanMany(&d2zj_, 1, &nj_, &nj_, 1, nj_, &njh, 1, njh, CUFFT_D2Z, dip_ * dkq_));
  int nih = ni_ / 2 + 1;
  CHECK(cufftPlanMany(&d2zi_, 1, &ni_, &ni_, 1, ni_, &nih, 1, nih, CUFFT_D2Z, dkq_ * djp_));
  #ifndef MPI_GPU
  CHECK(cudaHostAlloc(&ha_, bytes_ + bytes_, cudaHostAllocDefault));
  assert(ha_);
  hb_ = ha_ + nMax;
  #endif
}

PoissonZero3DBlockedGPU::~PoissonZero3DBlockedGPU()
{
  #ifndef MPI_GPU
  CHECK(cudaFreeHost(ha_));
  ha_ = hb_ = nullptr;
  #endif
  CHECK(cufftDestroy(d2zi_));
  CHECK(cufftDestroy(d2zj_));
  CHECK(cufftDestroy(d2zk_));
  MPI_Comm_free(&commI_);
  MPI_Comm_free(&commJ_);
  MPI_Comm_free(&commK_);
}

void print(const char *const title, const int ni, const int nj, const int nk, const double *const v)
{
  printf("%s:\n", title);
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      for (int k = 0; k < nk; k++) {
        printf("%.6f ", v[(i * nj + j) * nk + k]);
      }
      printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}

void PoissonZero3DBlockedGPU::solve(const long bytes, double *const density, double *const potential) const
{
  assert(bytes >= bytes_);
  assert(density);
  assert(potential);

  double *const ua             = potential;
  double *const ub             = density;
  cufftDoubleComplex *const uc = reinterpret_cast<cufftDoubleComplex *>(ub);

  const double ddi = ddi_;
  const double ddj = ddj_;
  const double ddk = ddk_;
  const int di     = di_;
  const int dip    = dip_;
  const int dj     = dj_;
  const int djp    = djp_;
  const int djq    = djq_;
  const int dk     = dk_;
  const int dkq    = dkq_;
  const int idi    = idi_;
  const int idj    = idj_;
  const int idp    = idp_;
  const int idq    = idq_;
  const int mp     = mp_;
  const int mq     = mq_;
  const int ni     = ni_;
  const int ni2    = ni2_;
  const int nj     = nj_;
  const int nj2    = nj2_;
  const int nk     = nk_;
  const int nk2    = nk2_;

  gpuFor(
      mp, mq, dip, djq, dk, GPU_LAMBDA(const int p, const int q, const int i, const int j, const int k) {
        const int iLo = p * dip;
        const int jLo = q * djq;
        if ((i + iLo < di) && (j + jLo < dj)) {
          ua[(((p * mq + q) * dip + i) * djq + j) * dk + k] = ub[((i + iLo) * dj + j + jLo) * dk + k];
        }
      });
  #ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_, dip * djq * dk, MPI_DOUBLE, hb_, dip * djq * dk, MPI_DOUBLE, commK_);
  CHECK(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
  #else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(ua, dip * djq * dk, MPI_DOUBLE, ub, dip * djq * dk, MPI_DOUBLE, commK_);
  #endif
  gpuFor(
      dip, djq, nk / 2 + 1, GPU_LAMBDA(const int i, const int j, const int k) {
        const int ij = (i * djq + j) * nk;
        const int kk = k + k;
        if (k == 0) {
          ua[ij] = ub[(i * djq + j) * dk];
        } else if (kk == nk) {
          const int pq  = (nk - 1) / dk;
          const int kpq = (nk - 1) % dk;
          ua[ij + k]    = -ub[((pq * dip + i) * djq + j) * dk + kpq];
        } else {
          const int pqa     = (kk - 1) / dk;
          const int kka     = (kk - 1) % dk;
          ua[ij + (nk - k)] = -ub[((pqa * dip + i) * djq + j) * dk + kka];
          const int pqb     = kk / dk;
          const int kkb     = kk % dk;
          ua[ij + k]        = ub[((pqb * dip + i) * djq + j) * dk + kkb];
        }
      });
  CHECK(cufftExecD2Z(d2zk_, ua, uc));
  gpuFor(
      dip, nk / 2 + 1, djq, GPU_LAMBDA(const int i, const int k, const int j) {
        if (k == 0) {
          const int q0                              = (nk - 1) / dkq;
          const int k0                              = (nk - 1) % dkq;
          ua[((q0 * dip + i) * dkq + k0) * djq + j] = 2.0 * ub[(i * djq + j) * nk2];
        } else if (k + k == nk) {
          const int qa                              = (nk / 2 - 1) / dkq;
          const int ka                              = (nk / 2 - 1) % dkq;
          ua[((qa * dip + i) * dkq + ka) * djq + j] = sqrt2 * ub[(i * djq + j) * nk2 + nk];
        } else {
          const int qa    = (nk - k - 1) / dkq;
          const int ka    = (nk - k - 1) % dkq;
          const int qb    = (k - 1) / dkq;
          const int kb    = (k - 1) % dkq;
          const double ak = 2.0 * ub[(i * djq + j) * nk2 + 2 * k];
          const double bk = 2.0 * ub[(i * djq + j) * nk2 + 2 * k + 1];
          double wa, wb;
          sincospi(double(k) / double(nk + nk), &wb, &wa);
          ua[((qa * dip + i) * dkq + ka) * djq + j] = wa * ak + wb * bk;
          ua[((qb * dip + i) * dkq + kb) * djq + j] = wb * ak - wa * bk;
        }
      });
  #ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_, dip * dkq * djq, MPI_DOUBLE, hb_, dip * dkq * djq, MPI_DOUBLE, commJ_);
  CHECK(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
  #else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(ua, dip * dkq * djq, MPI_DOUBLE, ub, dip * dkq * djq, MPI_DOUBLE, commJ_);
  #endif
  gpuFor(
      dip, dkq, nj / 2 + 1, GPU_LAMBDA(const int i, const int k, const int j) {
        const int ik = (i * dkq + k) * nj;
        if (j == 0) {
          ua[ik] = ub[(i * dkq + k) * djq];
        } else if (j + j == nj) {
          const int qa    = (nj - 1) / djq;
          const int ja    = (nj - 1) % djq;
          ua[ik + nj / 2] = -ub[((qa * dip + i) * dkq + k) * djq + ja];
        } else {
          const int qa    = (j + j - 1) / djq;
          const int ja    = (j + j - 1) % djq;
          ua[ik + nj - j] = -ub[((qa * dip + i) * dkq + k) * djq + ja];
          const int qb    = (j + j) / djq;
          const int jb    = (j + j) % djq;
          ua[ik + j]      = ub[((qb * dip + i) * dkq + k) * djq + jb];
        }
      });
  CHECK(cufftExecD2Z(d2zj_, ua, uc));
  gpuFor(
      dkq, nj / 2 + 1, dip, GPU_LAMBDA(const int k, const int j, const int i) {
        if (j == 0) {
          const int pa                              = (nj - 1) / djp;
          const int ja                              = (nj - 1) % djp;
          ua[((pa * dkq + k) * djp + ja) * dip + i] = 2.0 * ub[(i * dkq + k) * nj2];
        } else if (j + j == nj) {
          const int pa                              = (nj / 2 - 1) / djp;
          const int ja                              = (nj / 2 - 1) % djp;
          ua[((pa * dkq + k) * djp + ja) * dip + i] = sqrt2 * ub[(i * dkq + k) * nj2 + nj];
        } else {
          const double aj = 2.0 * ub[(i * dkq + k) * nj2 + 2 * j];
          const double bj = 2.0 * ub[(i * dkq + k) * nj2 + 2 * j + 1];
          double wa, wb;
          sincospi(double(j) / double(nj + nj), &wb, &wa);
          const int pa                              = (nj - j - 1) / djp;
          const int ja                              = (nj - j - 1) % djp;
          const int pb                              = (j - 1) / djp;
          const int jb                              = (j - 1) % djp;
          ua[((pa * dkq + k) * djp + ja) * dip + i] = wa * aj + wb * bj;
          ua[((pb * dkq + k) * djp + jb) * dip + i] = wb * aj - wa * bj;
        }
      });
  #ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_, dkq * djp * dip, MPI_DOUBLE, hb_, dkq * djp * dip, MPI_DOUBLE, commI_);
  CHECK(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
  #else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(ua, dkq * djp * dip, MPI_DOUBLE, ub, dkq * djp * dip, MPI_DOUBLE, commI_);
  #endif
  gpuFor(
      dkq, djp, ni / 2 + 1, GPU_LAMBDA(const int k, const int j, const int i) {
        const int kj = (k * djp + j) * ni;
        if (i == 0) {
          ua[kj] = ub[(k * djp + j) * dip];
        } else if (i + i == ni) {
          const int ida   = (ni - 1) / di;
          const int pa    = (ni - 1) % di / dip;
          const int ia    = ni - 1 - ida * di - pa * dip;
          ua[kj + ni / 2] = -ub[(((ida * mp + pa) * dkq + k) * djp + j) * dip + ia];
        } else {
          const int ida   = (i + i - 1) / di;
          const int pa    = (i + i - 1) % di / dip;
          const int ia    = i + i - 1 - ida * di - pa * dip;
          ua[kj + ni - i] = -ub[(((ida * mp + pa) * dkq + k) * djp + j) * dip + ia];
          const int idb   = (i + i) / di;
          const int pb    = (i + i) % di / dip;
          const int ib    = i + i - idb * di - pb * dip;
          ua[kj + i]      = ub[(((idb * mp + pb) * dkq + k) * djp + j) * dip + ib];
        }
      });
  CHECK(cufftExecD2Z(d2zi_, ua, uc));
  {
  #ifdef PARIS_GALACTIC_3PT
    const double si  = M_PI / double(ni + ni);
    const double sj  = M_PI / double(nj + nj);
    const double sk  = M_PI / double(nk + nk);
    const double iin = sqr(sin(double(ni) * si) * ddi);
  #elif defined PARIS_GALACTIC_5PT
    const double si  = M_PI / double(ni);
    const double sj  = M_PI / double(nj);
    const double sk  = M_PI / double(nk);
    const double cin = cos(double(ni) * si);
    const double iin = ddi * (2.0 * cin * cin - 16.0 * cin + 14.0);
  #else
    const double iin = sqr(double(ni) * ddi);
  #endif
    const int jLo = (idi * mp + idp) * djp;
    const int kLo = (idj * mq + idq) * dkq;
    gpuFor(
        dkq, djp, ni / 2 + 1, GPU_LAMBDA(const int k, const int j, const int i) {
          const int kj  = (k * djp + j) * ni;
          const int kj2 = (k * djp + j) * ni2;
  #ifdef PARIS_GALACTIC_3PT
          const double jjkk = sqr(sin(double(jLo + j + 1) * sj) * ddj) + sqr(sin(double(kLo + k + 1) * sk) * ddk);
  #elif defined PARIS_GALACTIC_5PT
          const double cj   = cos(double(jLo + j + 1) * sj);
          const double jj   = ddj * (2.0 * cj * cj - 16.0 * cj + 14.0);
          const double ck   = cos(double(kLo + k + 1) * sk);
          const double kk   = ddk * (2.0 * ck * ck - 16.0 * ck + 14.0);
          const double jjkk = jj + kk;
  #else
          const double jjkk =
              sqr(double(jLo + j + 1) * ddj) + sqr(double(kLo + k + 1) * ddk);
  #endif
          if (i == 0) {
            ua[kj] = -2.0 * ub[kj2] / (iin + jjkk);
          } else {
  #ifdef PARIS_GALACTIC_3PT
            const double ii = sqr(sin(double(i) * si) * ddi);
  #elif defined PARIS_GALACTIC_5PT
            const double ci = cos(double(i) * si);
            const double ii = ddi * (2.0 * ci * ci - 16.0 * ci + 14.0);
  #else
            const double ii = sqr(double(i) * ddi);
  #endif
            if (i + i == ni) {
              ua[kj + ni / 2] = -2.0 * ub[kj2 + ni] / (ii + jjkk);
            } else {
              const double ai = 2.0 * ub[kj2 + 2 * i];
              const double bi = 2.0 * ub[kj2 + 2 * i + 1];
              double wa, wb;
              sincospi(double(i) / double(ni + ni), &wb, &wa);
  #ifdef PARIS_GALACTIC_3PT
              const double nii = sqr(sin(double(ni - i) * si) * ddi);
  #elif defined PARIS_GALACTIC_5PT
              const double cni = cos(double(ni - i) * si);
              const double nii = ddi * (2.0 * cni * cni - 16.0 * cni + 14.0);
  #else
              const double nii = sqr(double(ni - i) * ddi);
  #endif
              const double aai = -(wa * ai + wb * bi) / (nii + jjkk);
              const double bbi = (wa * bi - wb * ai) / (ii + jjkk);
              const double apb = aai + bbi;
              const double amb = aai - bbi;
              ua[kj + i]       = wa * amb + wb * apb;
              ua[kj + ni - i]  = wa * apb - wb * amb;
            }
          }
        });
  }
  CHECK(cufftExecD2Z(d2zi_, ua, uc));
  gpuFor(
      dkq, ni / 2 + 1, djp, GPU_LAMBDA(const int k, const int i, const int j) {
        if (i == 0) {
          ua[k * dip * djp + j] = ub[(k * djp + j) * ni2];
        } else if (i + i == ni) {
          const int ida                                          = (ni - 1) / di;
          const int pa                                           = (ni - 1) % di / dip;
          const int ia                                           = ni - 1 - ida * di - pa * dip;
          ua[(((ida * mp + pa) * dkq + k) * dip + ia) * djp + j] = -ub[(k * djp + j) * ni2 + ni];
        } else {
          const double ai                                        = ub[(k * djp + j) * ni2 + i + i];
          const double bi                                        = ub[(k * djp + j) * ni2 + i + i + 1];
          const int ida                                          = (i + i - 1) / di;
          const int pa                                           = (i + i - 1) % di / dip;
          const int ia                                           = i + i - 1 - ida * di - pa * dip;
          ua[(((ida * mp + pa) * dkq + k) * dip + ia) * djp + j] = bi - ai;
          const int idb                                          = (i + i) / di;
          const int pb                                           = (i + i) % di / dip;
          const int ib                                           = i + i - idb * di - pb * dip;
          ua[(((idb * mp + pb) * dkq + k) * dip + ib) * djp + j] = ai + bi;
        }
      });
  #ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_, dkq * djp * dip, MPI_DOUBLE, hb_, dkq * djp * dip, MPI_DOUBLE, commI_);
  CHECK(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
  #else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(ua, dkq * djp * dip, MPI_DOUBLE, ub, dkq * djp * dip, MPI_DOUBLE, commI_);
  #endif
  gpuFor(
      dkq, dip, nj / 2 + 1, GPU_LAMBDA(const int k, const int i, const int j) {
        const long ki = (k * dip + i) * nj;
        if (j == 0) {
          const int pa = (nj - 1) / djp;
          const int ja = (nj - 1) - pa * djp;
          ua[ki]       = ub[((pa * dkq + k) * dip + i) * djp + ja];
        } else if (j + j == nj) {
          const int pa    = (nj / 2 - 1) / djp;
          const int ja    = nj / 2 - 1 - pa * djp;
          ua[ki + nj / 2] = sqrt2 * ub[((pa * dkq + k) * dip + i) * djp + ja];
        } else {
          const int pa     = (nj - 1 - j) / djp;
          const int ja     = nj - 1 - j - pa * djp;
          const double aj  = ub[((pa * dkq + k) * dip + i) * djp + ja];
          const int pb     = (j - 1) / djp;
          const int jb     = j - 1 - pb * djp;
          const double bj  = ub[((pb * dkq + k) * dip + i) * djp + jb];
          const double apb = aj + bj;
          const double amb = aj - bj;
          double wa, wb;
          sincospi(double(j) / double(nj + nj), &wb, &wa);
          ua[ki + j]      = wa * amb + wb * apb;
          ua[ki + nj - j] = wa * apb - wb * amb;
        }
      });
  CHECK(cufftExecD2Z(d2zj_, ua, uc));
  gpuFor(
      dip, nj / 2 + 1, dkq, GPU_LAMBDA(const int i, const int j, const int k) {
        if (j == 0) {
          ua[i * djq * dkq + k] = ub[(k * dip + i) * nj2];
        } else if (j + j == nj) {
          const int ida                                          = (nj - 1) / dj;
          const int qa                                           = (nj - 1) % dj / djq;
          const int ja                                           = nj - 1 - ida * dj - qa * djq;
          ua[(((ida * mq + qa) * dip + i) * djq + ja) * dkq + k] = -ub[(k * dip + i) * nj2 + nj];
        } else {
          const int jj                                           = j + j;
          const int ida                                          = (jj - 1) / dj;
          const int qa                                           = (jj - 1) % dj / djq;
          const int ja                                           = jj - 1 - ida * dj - qa * djq;
          const int idb                                          = jj / dj;
          const int qb                                           = jj % dj / djq;
          const int jb                                           = jj - idb * dj - qb * djq;
          const double aj                                        = ub[(k * dip + i) * nj2 + jj];
          const double bj                                        = ub[(k * dip + i) * nj2 + jj + 1];
          ua[(((ida * mq + qa) * dip + i) * djq + ja) * dkq + k] = bj - aj;
          ua[(((idb * mq + qb) * dip + i) * djq + jb) * dkq + k] = aj + bj;
        }
      });
  #ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_, dip * djq * dkq, MPI_DOUBLE, hb_, dip * djq * dkq, MPI_DOUBLE, commJ_);
  CHECK(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
  #else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(ua, dip * djq * dkq, MPI_DOUBLE, ub, dip * djq * dkq, MPI_DOUBLE, commJ_);
  #endif
  gpuFor(
      dip, djq, nk / 2 + 1, GPU_LAMBDA(const int i, const int j, const int k) {
        const long ij = (i * djq + j) * nk;
        if (k == 0) {
          const int qa = (nk - 1) / dkq;
          const int ka = nk - 1 - qa * dkq;
          ua[ij]       = ub[((qa * dip + i) * djq + j) * dkq + ka];
        } else if (k + k == nk) {
          const int qa    = (nk / 2 - 1) / dkq;
          const int ka    = nk / 2 - 1 - qa * dkq;
          ua[ij + nk / 2] = sqrt2 * ub[((qa * dip + i) * djq + j) * dkq + ka];
        } else {
          const int qa     = (nk - 1 - k) / dkq;
          const int ka     = nk - 1 - k - qa * dkq;
          const double ak  = ub[((qa * dip + i) * djq + j) * dkq + ka];
          const int qb     = (k - 1) / dkq;
          const int kb     = k - 1 - qb * dkq;
          const double bk  = ub[((qb * dip + i) * djq + j) * dkq + kb];
          const double apb = ak + bk;
          const double amb = ak - bk;
          double wa, wb;
          sincospi(double(k) / double(nk + nk), &wb, &wa);
          ua[ij + k]      = wa * amb + wb * apb;
          ua[ij + nk - k] = wa * apb - wb * amb;
        }
      });
  CHECK(cufftExecD2Z(d2zk_, ua, uc));
  const double divN = 1.0 / (8.0 * double(ni) * double(nj) * double(nk));
  gpuFor(
      dip, djq, nk / 2 + 1, GPU_LAMBDA(const int i, const int j, const int k) {
        if (k == 0) {
          ua[(i * djq + j) * dk] = divN * ub[(i * djq + j) * nk2];
        } else if (k + k == nk) {
          const int pqa                             = (nk - 1) / dk;
          const int ka                              = nk - 1 - pqa * dk;
          ua[((pqa * dip + i) * djq + j) * dk + ka] = -divN * ub[(i * djq + j) * nk2 + nk];
        } else {
          const int kk                              = k + k;
          const double ak                           = ub[(i * djq + j) * nk2 + kk];
          const double bk                           = ub[(i * djq + j) * nk2 + kk + 1];
          const int pqa                             = (kk - 1) / dk;
          const int ka                              = kk - 1 - pqa * dk;
          ua[((pqa * dip + i) * djq + j) * dk + ka] = divN * (bk - ak);
          const int pqb                             = kk / dk;
          const int kb                              = kk - pqb * dk;
          ua[((pqb * dip + i) * djq + j) * dk + kb] = divN * (ak + bk);
        }
      });
  #ifndef MPI_GPU
  CHECK(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
  MPI_Alltoall(ha_, dip * djq * dk, MPI_DOUBLE, hb_, dip * djq * dk, MPI_DOUBLE, commK_);
  CHECK(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
  #else
  CHECK(cudaDeviceSynchronize());
  MPI_Alltoall(ua, dip * djq * dk, MPI_DOUBLE, ub, dip * djq * dk, MPI_DOUBLE, commK_);
  #endif
  gpuFor(
      mp, dip, mq, djq, dk, GPU_LAMBDA(const int p, const int i, const int q, const int j, const int k) {
        const int iLo = p * dip;
        const int jLo = q * djq;
        if ((iLo + i < di) && (jLo + j < dj)) {
          ua[((i + iLo) * dj + j + jLo) * dk + k] = ub[(((p * mq + q) * dip + i) * djq + j) * dk + k];
        }
      });
}

#endif
