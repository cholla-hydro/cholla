/*! \file RT.h
 *  \brief Declarations for the radiative transfer functions */

#ifdef RT

#ifndef RT_H
#define RT_H

#include"../global.h"


class Rad3D
{
  public:

  // Struct for storing abundance data
  RT_field_data abundances;

  // constructor
  Rad3D( void );

  void Initialize( struct parameters *P);

  void Allocate_Memory_RT();

  void Free_Memory(void);

};

#endif
#endif //RT
