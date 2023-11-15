#include <string>
#include <vector>

#include "../io/io.h"
#include "../feedback/s99table.h"
#include "../feedback/ratecalc.h"



feedback::SNRateCalc::SNRateCalc(struct parameters& P)
  : SNRateCalc() // the dfault constructor sets up some sensible defaults
{

  chprintf("feedback::Init_State start\n");
  std::string snr_filename(P.snr_filename);
  if (snr_filename.empty()) {
    // the constant-rate configuration is handled by the default constructor for us
    chprintf("No SN rate file specified.  Using constant rate\n");
  } else {
    chprintf("Specified a SNR filename %s.\n", snr_filename.data());

    feedback::S99Table tab = parse_s99_table(snr_filename, feedback::S99TabKind::supernova);
    const std::size_t time_col = tab.col_index("TIME");
    const std::size_t rate_col = tab.col_index("ALL SUPERNOVAE: TOTAL RATE");
    const std::size_t nrows = tab.nrows();

    // read in array of supernova rate values.
    std::vector<Real> snr_time(nrows);
    std::vector<Real> snr(nrows);

    for (std::size_t i = 0; i < nrows; i++){
      // in the following divide by # years per kyr (1000)
      snr_time[i] = tab(time_col, i) / 1000;
      snr[i] = pow(10,tab(rate_col, i)) * 1000 / S_99_TOTAL_MASS;
    }

    time_sn_end_   = snr_time[snr_time.size() - 1];
    time_sn_start_ = snr_time[0];
    // the following is the time interval between data points
    // (i.e. assumes regular temporal spacing)
    snr_dt_ = (time_sn_end_ - time_sn_start_) / (snr.size() - 1);

    CHECK(cudaMalloc((void**)&dev_snr_, snr.size() * sizeof(Real)));
    CHECK(cudaMemcpy(dev_snr_, snr.data(), snr.size() * sizeof(Real), cudaMemcpyHostToDevice));

  }

}

/* Read in Stellar wind data from Starburst 99. If no file exists, assume a
 * constant rate.
 *
 *
 * @param P reference to parameters struct. Passes in starburst 99 filepath
 */
feedback::SWRateCalc::SWRateCalc(struct parameters& P)
  : dev_sw_p_(nullptr),
    dev_sw_e_(nullptr),
    sw_dt_(0.0),
    time_sw_start_(0.0),
    time_sw_end_(0.0)
{
#if (!defined(FEEDBACK)) || defined(NO_WIND_FEEDBACK)
  return;
#else
  chprintf("Init_Wind_State start\n");
  std::string sw_filename(P.sw_filename);
  if (sw_filename.empty()) {
    CHOLLA_ERROR("must specify a stellar wind file.\n");
  }

  feedback::S99Table tab = parse_s99_table(sw_filename, feedback::S99TabKind::stellar_wind);

  const std::size_t COL_TIME       = tab.col_index("TIME");
  const std::size_t COL_POWER      = tab.col_index("POWER: ALL");
  const std::size_t COL_ALL_P_FLUX = tab.col_index("MOMENTUM FLUX: ALL");
  const std::size_t nrows          = tab.nrows();

  std::vector<Real> sw_time(nrows);
  std::vector<Real> sw_p(nrows);
  std::vector<Real> sw_e(nrows);

  for (std::size_t i = 0; i < nrows; i++){
    sw_time[i] = tab(COL_TIME, i) / 1000;  // divide by # years per kyr (1000)
    sw_e[i]    = tab(COL_POWER, i);
    sw_p[i]    = tab(COL_ALL_P_FLUX, i);
  }

  time_sw_end_   = sw_time[sw_time.size() - 1];
  time_sw_start_ = sw_time[0];
  // the following is the time interval between data points
  // (i.e. assumes regular temporal spacing)
  sw_dt_ = (time_sw_end_ - time_sw_start_) / (sw_p.size() - 1);
  chprintf("wind t_s %.5e, t_e %.5e, delta T %0.5e\n", time_sw_start_, time_sw_end_, sw_dt_);

  CHECK(cudaMalloc((void**)&dev_sw_p_, sw_p.size() * sizeof(Real)));
  CHECK(cudaMemcpy(dev_sw_p_, sw_p.data(), sw_p.size() * sizeof(Real), cudaMemcpyHostToDevice));

  CHECK(cudaMalloc((void**)&dev_sw_e_, sw_e.size() * sizeof(Real)));
  CHECK(cudaMemcpy(dev_sw_e_, sw_e.data(), sw_e.size() * sizeof(Real), cudaMemcpyHostToDevice));

  chprintf("first 40 stellar wind momentum values:\n");
  for (int i = 0; i < 40; i++) {
    chprintf("%0.5e  %5f %5f \n", sw_time.at(i), sw_e.at(i), sw_p.at(i));
  }
#endif
}
