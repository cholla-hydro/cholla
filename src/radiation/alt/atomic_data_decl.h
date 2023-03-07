/*LICENSE*/

#ifndef PHYSICS_ATOMIC_DATA_DECL_ANY_H
#define PHYSICS_ATOMIC_DATA_DECL_ANY_H

#include "align.h"

//
//  Various atomic, chemical, and thermal rates.
//  This is just declarations for indices and convenience wrappers.
//
template <typename value_t, unsigned int N>
class StaticTable;
template <typename value_t, unsigned int N, char Mode>
class StaticTableGPU;

namespace Physics
{
namespace AtomicData
{
constexpr double TionHI   = 157807.0;  // ionization threshold in K
constexpr double TionHeI  = 285335.0;  // ionization threshold in K
constexpr double TionHeII = 631515.0;  // ionization threshold in K

constexpr double Ry_eV =
    13.605693122994;  // Rydberg in eV, violating the style here, but all other forms look more confusing

//
//  A more complex object that repackages cross-sections differently from a simple table.
//  It is more convenient to use in integrating photo-ionization rates.
//
struct DEVICE_ALIGN_DECL CrossSection {
  enum : unsigned int { IonizationHI, IonizationHeI, IonizationHeII, IonizationCVI, END };
  static constexpr unsigned int Num = END;

  struct DEVICE_ALIGN_DECL Threshold {
    unsigned int idx;
    float xi;
    float hnu_K;   // in K, violating the style here, but all other forms look more confusing
    float hnu_eV;  // in eV
    float cs;      // in barns (to limit the numeric range and fit into float).
  };

  float *xi;       // xi = log(hnu/1Ry)
  float *hnu_K;    // in K, violating the style here, but all other forms look more confusing
  float *hnu_eV;   // in eV
  float *cs[Num];  // in barns (to limit the numeric range and fit into float).

  Threshold thresholds[Num];

  //
  //  Some cached values that are used often
  //
  float csHIatHI;
  float csHIatHeI;
  float csHIatHeII;
  float csHeIatHeI;
  float csHeIatHeII;
  float csHeIIatHeII;

  unsigned int nxi;
  float xiMin, xiMax, dxi;
};
};  // namespace AtomicData
};  // namespace Physics

#endif  // PHYSICS_ATOMIC_DATA_DECL_ANY_H
