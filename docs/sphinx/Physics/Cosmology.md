# Cosmology

:::{todo}
Add documentation for cosmology. A description of the cosmology implementation can be found in [Villasenor et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...912..138V/abstract).
:::

## Expansion History

At each time step, we calculate the Hubble parameter

```{math}
\left(\frac{H(z)}{H_0}\right)^2 = \Omega_{m,0} (1 + z)^{-3} + \Omega_{R,0} (1 + z)^{-4} + \Omega_{k,0} (1 + z)^{-2} + \Omega_{DE,0} \frac{\rho_{DE}(z)}{\rho_{DE}(z=0)}
```

where $H_0$ is the present-day Hubble parameter, $\Omega_{m,0}$ is the present-day matter energy-density, $\Omega_{R,0}$ is the present-day radiation energy-density, the energy-density related to the curvature of the Universe is $\Omega_{k,0}$, and $\Omega_{DE,0}$ is the present-day dark energy energy-density. To ensure a flat cosmology, we set $\Omega_{k,0} = 1 - \Omega_{m,0} - \Omega_{R,0} - \Omega_{DE,0}$.

When {par:param}`wDE_file` is specified, we calculate a table for the dark energy contribution as

```{math}
\frac{\rho_{DE}(z)}{\rho_{DE}(z=0)} = (1 + z)^3 \exp\left[3 \int_{z=0}^z \frac{w(z')}{1+z'} dz'  \right]
```

using a midpoint rule integration method. For a {math}`w_0 w_a CDM` cosmology, this factor is calculated as

```{math}
\frac{\rho_{DE}(z)}{\rho_{DE}(z=0)} = (1 + z)^{3(1 + w_0 + w_a)} \exp\left[ \frac{-3 w_a z}{1+z}  \right]
```

where {math}`w(a) = w_0 + w_a (1 - a)`. By default, we assume a $\Lambda$CDM cosmology where $(w_0,w_a) = (-1,0)$, and {math}`\rho_{DE}(z) / \rho_{DE}(z=0) = 1`.


## Cosmological Initial Conditions

Cholla now includes routines to generate initial conditions to be used in running cosmological simulations.  The scheme follows the method described by [Hahn et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503..426H/abstract), but is currently only first-order (Zel'dovich) accurate.

The users must specify cosmological parameters, and provide a 3-column power spectrum file that includes (1) wavenumber in h/Mpc, (2) P(k) in (Mpc/h)^3, and (3) the baryon-dark matter relative fluctuation power spectrum in (Mpc/h)^3.  The initial gas temperature and ionization state must also be specified, along with a random number generation seed.

The ICs generator first draws white-noise normal random numbers across the grid. The noise field is filtered with the power spectrum in Fourier space, and then transformed back to generate the initial overdensity field. The overdensity field is used to compute the initial potential by solving Poisson's equation in Fourier space again. Then finite differences are used to compute the displacements of the dark matter and the velocity field. The deformation tensore is computed from a second derivative of the potential field, again using finite-difference methods, and then used to set the gas density field.  The gas temperature is set by the parameter file, and then the gas energies and momentum fields are computed. Particles are exchanged as needed between subvolumes, and then the simulation begins.


Here is an example parameter file for a 8192^3 simulation:

```
######################################
#
# Parameter File for a 200 Mpc/h 8192^3 sim
#
######################################
# number of grid cells in the x dimension
nx=8192
# number of grid cells in the y dimension
ny=8192
# number of grid cells in the z dimension
nz=8192
# output time
tout=1000
# how often to output
outstep=1000
#n_steps_output=10
# value of gamma
gamma=1.66666667
# name of initial conditions
init=Cosmological_ICs
nfile=0
scale_outputs_file=outputs_cosmo_z_100_2_2.txt
# domain properties
xmin=0.0
ymin=0.0
zmin=0.0
xlen=200000.0
ylen=200000.0
zlen=200000.0
# type of boundary conditions
xl_bcnd=1
xu_bcnd=1
yl_bcnd=1
yu_bcnd=1
zl_bcnd=1
zu_bcnd=1
# path to output directory
indir=ics/
outdir=data/
UVB_rates_file=uvb_rates_V22.txt
#Generated from planck_2018.ini
#sigma_8 = 0.81183
#Hubble parameter in km/s/Mpc
H0=67.32117
#Fractional matter critical density
Omega_M=0.314400
#Fractional baryon critical density
Omega_b=0.049387
#Fractional DE critical density
Omega_L=0.685508
#Fractional radiation critical density
Omega_R=9.231186e-05
#DE equation of state parameter w0
w0=-1.000000e+00
#DE equation of state parameter wa
wa=0
#Initial hydrogen ionization fraction
xHp=3.737205e-04
#Initial helium single ionization fraction
xHep=9.977809e-16
#Initial gas temperature from RECFAST
Tgas=6.196050e+02
#Initial helium mass fraction
YHe=2.454006e-01
#Initial redshift
Init_redshift=2.500000e+02
#cosmological power spectrum file
cosmo_ics_pk_file=power_spectrum.planck_2018.txt
#RNG seed
seed=1337
```