/*! \file
 * This file defines \ref TabulatedDynamicalDarkEnergyEoS
 */

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../global/global.h"
#include "../utils/error_handling.h"

class TabulatedDynamicalDarkEnergyEoS
{
 private:
  std::vector<float> dynamicalDE_table_z;
  std::vector<float> dynamicalDE_table_w;

  // dynamical dark energy energy density normalized to the present-date rho_DE(z) / rho_DE(z=0)
  std::vector<float> dynamicalDE_table_density;

  /*! Load redshift and dark energy equation of state table z, wDE(z), only called once to setup dynamical DE case */
  void Setup_DynamicalDE_EquationOfState_(std::istream& in, const std::string& fname, bool silent);

  /*! Calculate dark energy density normalized to z=0, populate dynamicalDE_table_density */
  void Set_DynamicalDE_Density();

  void Setup_Full(std::istream& in, const std::string& fname) {}

 public:
  /*! Interpolate dynamicalDE_table_density to find rhoDE(z) / rhoDE(z=0) at z=1/a - 1 */
  Real Get_DynamicalDE_Density_from_a(Real a);

  /*! Construct a new instance
   *
   *  By default, when \p f is a ``nulllptr, this function tries to open the file named
   *  \p path. Otherwise, this function treats \p f as if it's a newly openned stream
   *  associated with \p path (this secondary behavior is useful for testing purposes).
   */
  explicit TabulatedDynamicalDarkEnergyEoS(const std::string& path, std::istream* f = nullptr, bool silent = false)
  {
    std::fstream tmp;
    if (f == nullptr) {
      tmp.open(path, std::ios_base::in);
      CHOLLA_ASSERT(tmp.is_open(), "Unable to open DE equation of state file: %s\n", path.c_str());
      f = dynamic_cast<std::istream*>(&tmp);
    }
    Setup_DynamicalDE_EquationOfState_(*f, path, silent);
    Set_DynamicalDE_Density();
  }
};
