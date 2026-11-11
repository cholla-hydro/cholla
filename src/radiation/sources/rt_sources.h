/*! \file rt_sources.h
 *  \brief Declarations of the RTSources class. */

#ifdef RT

#ifndef RT_SOURCES_H
#define RT_SOURCES_H

#ifdef MPI_CHOLLA
  #include "../mpi/mpi_routines.h"
#endif /*MPI_CHOLLA*/

#include <stdio.h>

#include <functional>

#include "../global/global.h"  // declares Parameter and forward-declares ParameterMap
#include "../global/global_cuda.h"
#include "../grid/field_info.h"
#include "../io/FnameTemplate.h"
#include "../model/model_collection.h"

#ifdef HDF5
  #include <hdf5.h>
#endif

#ifdef GRAVITY
  #include "../gravity/grav3D.h"
#endif

#ifdef PARTICLES
  #include "../particles/particles_3D.h"
#endif

#ifdef COSMOLOGY
  #include "../cosmology/cosmology.h"
  #include "../cosmology/rng.h"
#endif

#ifdef FFT
  #include "../fft/fft_3D.h"
#endif

#ifdef RT
  #include "../radiation/radiation.h"
#endif

#ifdef CHEMISTRY_GPU
  #include "chemistry_gpu/chemistry_gpu.h"
#endif

struct RT_State {

  /*! \var t
   *  \brief Current simulation time in code units. */
  Real t;

  /*! \var dt 
   *  \brief Current simulation time step in code units. */
  Real dt;

  /*! \var current_a
   *  \brief Current cosmological scale factor. */
  Real current_a;

  /*! \var dx 
   *  \brief Current x cell width. */
  Real dx;

  /*! \var dy 
   *  \brief Current y cell width. */
  Real dy;

  /*! \var dz
   *  \brief Current z cell width. */
  Real dz;

  /*! \var xbound */
  /*  \brief Global domain x-direction minimum */
  Real xbound;

  /*! \var ybound */
  /*  \brief Global domain y-direction minimum */
  Real ybound;

  /*! \var zbound */
  /*  \brief Global domain z-direction minimum */
  Real zbound;

  /*! \var xblocal */
  /*  \brief Local domain x-direction minimum */
  Real xblocal;

  /*! \var yblocal */
  /*  \brief Local domain y-direction minimum */
  Real yblocal;

  /*! \var zblocal */
  /*  \brief Local domain z-direction minimum */
  Real zblocal;

  /*! \var xblocal_max */
  /*  \brief Local domain x-direction maximum */
  Real xblocal_max;

  /*! \var yblocal_max */
  /*  \brief Local domain y-direction maximum */
  Real yblocal_max;

  /*! \var zblocal_max */
  /*  \brief Local domain z-direction maximum */
  Real zblocal_max;

} rt_state;

class RTSourceModel {

  public:

    /*! \fn RTSourceModel(void)
     * \brief Constructor for RT Source Model */
    RTSourceModel(void);

    /*! \fn void Initialize(const Parameters &P, RT_State state)
     *  \brief Initializes the source model */
    void Initialize(const Parameters &P, RT_State state);

    /*! \fn void Update(RT_State state) 
     *  \brief  Update the source properties */
    void Update(RT_State state);

};

#endif // RT_SOURCES_H
#endif // RT