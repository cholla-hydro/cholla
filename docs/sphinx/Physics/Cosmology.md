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


