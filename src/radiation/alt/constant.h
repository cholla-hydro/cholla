/*LICENSE*/

#ifndef CONSTANT_H
#define CONSTANT_H

namespace Constant
{
//
// Base units (CGS)
//
constexpr double K     = 1;
constexpr double cms   = 1;
constexpr double erg   = 1;
constexpr double barye = 1;
constexpr double dyne  = 1;
constexpr double g_cc  = 1;
constexpr double cc    = 1;

//
//  NG: values for pc and GMsun are from http://ssd.jpl.nasa.gov/?constants
//      values for mp, G, k, c, eV, amu are from Particle Physics Booklet 2008
//
constexpr double yr  = 365.25 * 86400; /* Julian year in seconds */
constexpr double Myr = 1.0e6 * yr;
constexpr double Gyr = 1.0e9 * yr;

constexpr double pc  = 3.0856775813e18;
constexpr double kpc = 1.0e3 * pc;
constexpr double Mpc = 1.0e6 * pc;

constexpr double km_s = 1.0e5;  // cm/s

constexpr double mp = 1.672621637e-24;
constexpr double mb = 1.670673249e-24;
constexpr double kB = 1.3806504e-16;
constexpr double G  = 6.67428e-8;
constexpr double c  = 2.99792458e10;

constexpr double eV     = 1.602176487e-12;
constexpr double sigmaT = 6.6524e-25;  // Thompson cross-section

constexpr double Msun = 1.32712440018e26 / G;
constexpr double Lsun = 3.839e33;  // from Wikipedia
};                                 // namespace Constant

#endif  // CONSTANT_H
