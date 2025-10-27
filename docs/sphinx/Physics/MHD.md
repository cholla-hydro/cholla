# MHD

The magnetohydrodynamics (MHD) implementation utilizes the VL+CT integrator described in [Stone & Gardiner 2009](https://ui.adsabs.harvard.edu/abs/2009NewA...14..139S/abstract) with the HLLD Riemann solver from [Miyoshi & Kusano 2005](https://ui.adsabs.harvard.edu/abs/2005JCoPh.208..315M/abstract); details of the Cholla implementation and performance can be found in Caddy & Schneider 2024. Interface reconstruction is supported in first, second, and third order with the following methods: Piecewise Constant Method (PCM), Piecewise Linear Method with limiting in the Characteristic Variables (PLMC)([Stone et al. 2005](https://ui.adsabs.harvard.edu/abs/2008ApJS..178..137S/abstract)), and Piecewise Parabolic Method in the Characteristic Variables (PPMC)([Felker & Stone 2018](https://ui.adsabs.harvard.edu/abs/2018JCoPh.375.1365F/abstract)).

The VL+CT integrator utilizes a staggered grid with the magnetic fields centered at cell faces rather than cell centers. All magnetic fields are stored at the i+1/2 interface.

## Turning on MHD & Supported Options

To turn on MHD, build with the `MHD` flag. In addition, MHD only supports the Van Leer (`VL`) integrator and the `HLLD` Riemann solver. MHD supports all reconstruction methods (`PCM`, `PLMP`, `PLMC`, and `PPMC`) except `PPMP` (PPM with limiting in the primitive variables). If Cholla MHD is built with an unsupported configuration, then it should raise a compile time error; if it does not, please open an issue about it.

### Output

MHD only supports output with HDF5 files, so if output is enabled, the `HDF5` flag must also be passed. Full grid and slice outputs are fully supported. In the case of slices, the magnetic fields are converted to cell-centered values. Projections and rotated projections also supported for outputting density, temperature, and in the case of rotated projections, velocity as well.

## Initializing

The initial magnetic fields can be set in the same manner as the hydro fields, see the `Grid3D::Orszag_Tang_Vortex` function in [`initial_conditions.cpp`](https://github.com/cholla-hydro/cholla/blob/main/src/grid/initial_conditions.cpp) for an example. Note that the initial conditions *must* be divergence-free for the VL+CT integrator to work. The easiest way to ensure this condition for non-trivial fields is to initialize from the vector potential rather than the magnetic field directly; the `mhd::utils::Init_Magnetic_Field_With_Vector_Potential` function in [`mhd_utilities.cu`](https://github.com/cholla-hydro/cholla/blob/main/src/grid/mhd_utilities.cu) has been provided to convert a vector potential into the magnetic field.

## Reconstruction Choice

Piecewise Linear Method (PLM) is the default. We have found that PPMC, while giving more accurate results in problems with smooth flows, is often very oscillatory near shocks. PLMC and PLMP give similar results, PLMP is slightly more oscillatory near shocks, though this is not present in all problems and is slightly faster than PLMC (~4-5%). Either PLM method works well, though one may work better than the other for specific problems.

## Caveats and Known Issues

MHD support has not yet been extended to all other physics modules. Support in other modules is a work in progress but as yet is not complete.
