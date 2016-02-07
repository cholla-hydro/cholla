#-------------------------------------------------------------------------------
# configure.ac: Athena configuration script template file.  Run 'autoconf'
# to generate a new 'configure' script from this file.
#
# When configure is run, it uses the command line options and default settings
# to create custom versions of the following files:
#    Makefile, Makeoption, src/Makefile, src/defs.h, src/*/Makefile,
#
# If new configure options are added to this file, they should also be
# added in show_config() in src/show_config.c, and to the diagnostic message
# output at the end of this file.
#
# This template adds the following options to the 'configure' script:
#
# PHYSICS "packages":
#   --with-problem='name'                 (use prob/name.c as problem generator)
#   --with-gas=[hydro,mhd]                              (hydro or mhd algorithm)
#   --with-eos=[isothermal, adiabatic]                       (equation of state)
#   --with-nscalars=n                               (number of advected scalars)
#   --with-gravity=[fft,fft_disk,fft_obc,multigrid]               (self-gravity)
#   --with-particles=[feedback,passive]              (dust particle integration)
#   --with-coord=[cartesian,cylindrical]                     (coordinate system)
#
# PHYSICS "features":
#   --enable-conduction                            (explicit thermal conduction)
#   --enable-resistivity                                  (explicit resistivity)
#   --enable-special-relativity              (special relativistic hydro or MHD)
#   --enable-viscosity                                      (explicit viscosity)
#
# ALGORITHM "packages":
#   --with-order=[1,2,3,2p,3p]        (order and type of spatial reconstruction)
#   --with-flux=[roe,hlle,hllc,hlld,force,exact,two-shock]       (flux function)
#   --with-integrator=[ctu,vl]                   (unsplit integration algorithm)
#   --with-cflags=[opt,debug,profile]                       (set compiler flags)
#
# ALGORITHM "features":
#   --enable-fargo                                      (enable FARGO algorithm)
#   --enable-fft                (compile and link with FFTW block decomposition)
#   --enable-fofc                 (first-order flux correction in VL integrator)
#   --enable-ghost                      (write out ghost cells in outputs/dumps)
#   --enable-h-correction              (turn on H-correction in multidimensions)
#   --enable-mpi                                          (parallelize with MPI)
#   --enable-shearing box                    (include shearing box source terms)
#   --enable-single                                 (double or single precision)
#   --enable-sts                     (super timestepping for explicit diffusion)
#   --enable-smr                                        (static mesh refinement)
#   --enable-rotating_frame                    (enable ROTATING_FRAME algorithm)
#   --enable-l1_inflow                             (enable inflow from L1 point)
#
#-------------------------------------------------------------------------------
# generic things

AC_INIT([cholla], [0.1], [eschneider@as.arizona.edu])
AC_PROG_CC
AC_PROG_INSTALL



#-------------------------------------------------------------------------------
# ALGORITHM PACKAGE: set the order and type of the spatial reconstruction
#   --with-order=[1,2,3] - #-order reconstruction in the characteristic vars
#   --with-order=[2p,3p] - #-order reconstruction in the primitive      vars
#  (default is 2 -- 2nd-order reconstruction in the characteristic vars)

AC_SUBST(ORDER)
AC_SUBST(RECONSTRUCTION)

AC_ARG_WITH(order,
	[--with-order=ORDER      Order of Accuracy (1,2,3,2p,3p)],
	with_order=$withval, with_order="3")
ORDER=$with_order
if test   "$with_order" = "1"; then
  RECONSTRUCTION="PCM"
elif test "$with_order" = "2"; then
  RECONSTRUCTION="PLMC"
elif test "$with_order" = "2p"; then
  RECONSTRUCTION="PLMP"
elif test "$with_order" = "3"; then
  RECONSTRUCTION="PPMC"
elif test "$with_order" = "3p"; then
  RECONSTRUCTION="PPMP"
else
  AC_MSG_ERROR([expected --with-order=1, 2, 3, 2p, or 3p])
fi

#-------------------------------------------------------------------------------
# ALGORITHM PACKAGE: flux function
#   --with-flux=[roe,hlle,hllc,hlld,force,exact]
#   (default is roe if gas=mhd, hllc if gas=hydro)

AC_SUBST(FLUX_NAME)
AC_SUBST(SOLVER)
AC_ARG_WITH(flux,
	[--with-flux=FLUX_NAME   Flux function (roe, exact)],
	with_flux=$withval, with_flux="roe")
FLUX_NAME=$with_flux
if test   "$with_flux" = "roe"; then
  SOLVER="ROE"
elif test "$with_flux" = "exact"; then
  SOLVER="EXACT"
else
  AC_MSG_ERROR([Invalid flux function, valid types are: roe, exact])
fi

#-------------------------------------------------------------------------------
# ALGORITHM PACKAGE: set the unsplit integration algorithm
#   --with-integrator=[ctu,vl] (default is ctu)

AC_SUBST(INTEGRATOR_NAME)
AC_SUBST(INTEGRATOR)

AC_ARG_WITH(integrator,
	[--with-integrator=INTEGRATOR_NAME  Which unsplit integrator (ctu, vl)],
	with_integrator=$withval, with_integrator="ctu")
INTEGRATOR_NAME=$with_integrator
if test   "$with_integrator" = "ctu"; then
  INTEGRATOR="CTU"
elif test "$with_integrator" = "vl"; then
  INTEGRATOR="VL"
else
  AC_MSG_ERROR([expected --with-integrator=ctu or vl])
fi


#-------------------------------------------------------------------------------
# ALGORITHM FEATURE: precision of floating point arithmetic
#   --enable-single (default is double)

AC_SUBST(PRECISION)
AC_ARG_ENABLE(single,
	[--enable-single  single-precision (default is double)], 
	ok=$enableval, ok=no)
if test "$ok" = "yes"; then
  PRECISION="PRECISION=1"
else
  PRECISION="PRECISION=2"
fi

#-------------------------------------------------------------------------------
# ALGORITHM FEATURE: write ghost cells in outputs/dumps
#   --enable-ghost
AC_SUBST(OUTPUT_NAME)
AC_SUBST(OUTPUT)

AC_ARG_WITH(output,
	[--with-output=OUTPUT_NAME Which output method (hdf5, binary)],
	with_output=$withval, with_output="hdf5")
OUTPUT_NAME=$with_output
if test   "$with_output" = "hdf5"; then
  OUTPUT="HDF5"
elif test "$with_integrator" = "vl"; then
  OUTPUT="BINARY"
else
  AC_MSG_ERROR([expected --with-output=hdf5 or binary])
fi


#-------------------------------------------------------------------------------
# ALGORITHM FEATURE: parallelize with MPI, --enable-mpi (default is no MPI)

AC_SUBST(MPI_MODE)
AC_ARG_ENABLE(mpi,
	[--enable-mpi  enable MPI parellelization],
	ok=$enableval, ok=no)
if test "$ok" = "yes"; then
  MPI_MODE="MPI_CHOLLA"
  MPI_MODE_USER="ON"
else
  MPI_MODE="NO_MPI_CHOLLA"
  MPI_MODE_USER="OFF"
fi

#-------------------------------------------------------------------------------
# ALGORITHM FEATURE: turn on H-correction in multidimensional integrators
#   --enable-h-correction
# Note that H-correction only works with Roe flux.  Prints error message if
# H-correction is enabled with any other flux.

AC_SUBST(H_CORRECTION_MODE)
AC_ARG_ENABLE(h-correction,
	[--enable-h-correction  turn on H-correction],
	ok=$enableval, ok=no)
if test "$ok" = "yes"; then
if test "$with_flux" = "roe"; then
  H_CORRECTION_MODE="H_CORRECTION"
  H_CORRECTION_MODE_USER="ON"
else
  AC_MSG_ERROR([H-correction only works with Roe flux])
fi
else
  H_CORRECTION_MODE="NO_H_CORRECTION"
  H_CORRECTION_MODE_USER="OFF"
fi


#-------------------------------------------------------------------------------
# ALGORITHM FEATURE: use cuda, --enable-cuda (default is no cuda)

AC_SUBST(CUDA_MODE)
AC_ARG_ENABLE(cuda,
	[--enable-cuda  enable CUDA code],
	ok=$enableval, ok=no)
if test "$ok" = "yes"; then
  CUDA_MODE="CUDA"
  CUDA_MODE_USER="ON"
else
  CUDA_MODE="NO_CUDA"
  CUDA_MODE_USER="OFF"
fi


#-------------------------------------------------------------------------------
# check for various library functions

AC_PROG_GCC_TRADITIONAL
AC_CHECK_FUNCS(strdup)
AC_C_BIGENDIAN

#-------------------------------------------------------------------------------
# date and time of configure, becomes a macro in defs.h

A_CONFIGURE_DATE="`date`"
AC_SUBST(A_CONFIGURE_DATE)

#-------------------------------------------------------------------------------
# write final diagnostic output

AC_SUBST(WARNING1)
AC_SUBST(WARNING2)
WARNING1="WARNING! This file has been automatically generated by configure."
WARNING2="Any changes to it will be overwritten the next time configure is run."

AC_CONFIG_FILES(Makefile Makeoptions src/Makefile)
AC_OUTPUT

echo ""
echo "Your Cholla distribution has now been configured:"
echo ""
echo "--- Algorithm Options ----"
echo ""
echo "Spatial Order:           $ORDER ($ACCURACY)"
echo "Flux:                    $FLUX_NAME"
echo "unsplit integrator:      $INTEGRATOR"
echo "Precision:               $PRECISION"
echo "Parallel modes: MPI      $MPI_MODE_USER"
echo "H-correction:            $H_CORRECTION_MODE_USER"
echo "CUDA:                    $CUDA_MODE_USER"

