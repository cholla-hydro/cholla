# Dual Energy Formalism

Cholla has the capability to employ the dual-energy formalism in order to simulate high Mach number flows.

## Brief Primer

To ensure the conservation of mass, momentum, and energy, the hydro solvers provided in Cholla evolve the conserved quantities {math}`\rho`, {math}`\rho v_x`, {math}`\rho v_y`, {math}`\rho v_z`, {math}`E`. The bulk of the work during each timestep is to calculate the fluxes in each conserved quantites between every cell. The flux calculations require knowledge of the thermal pressure, which is given by {math}`p = (\gamma - 1) (E-E_{\rm kinetic})`. 

Problems arise in simulations with large Mach numbers (such as cosmological simulations). When the Mach number is very large, then {math}`(E - E_{\rm kinetic})` is very small. Numerical issues arise, since the thermal pressure linearly scales with this small difference between two very large numbers.

## About the Dual Energy Formalism

The solution to this numerical problem is to use the "dual-energy formalism" (more details are provided in [Bryan+ 2014](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...19B)). The core idea is to track an extra separately-advected "thermal energy" field at each cell-location, in addition to the total energy field and use this "thermal energy" field in cases where {math}`(E - E_{\rm thermal})` provides insufficient precision. 

The dual-energy formalism is parameterized by two parameters, {math}`\eta_1` and {math}`\eta_2`. It's easiest to understand their meaning by discussing how they are used. The [Bryan+ 2014](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...19B) paper describes two main steps:
1. During a given timestep, when we want to compute thermal pressure, we compare quotient of the "thermal energy" field divided by {math}`E` to {math}`\eta_1`.
    - When the ratio is smaller than {math}`\eta_1` we use the "thermal energy" field. When it exceeds {math}`\eta_1`, we use {math}`(E-E_{\rm kinetic})`.
    - In effect, {math}`\eta_1` directly parameterizes the precision where the dual-energy formalism kicks in.
    - It's worth mentioning that running a simulation with {math}`\eta_1=0` is equivalent to running a simulation without the dual energy formalism.
2. Near the end of each timestep (after updating all the fields with fluxes and any source terms), the "thermal energy" energy is optionally overwritten with the value taken from {math}`(E-E_{\rm kinetic})`.
    - To motivate this step, it's important to understand that when we separately advect the "thermal energy" and add the {math}`-p(\nabla \cdot {\bf v})\Delta t/ \rho` source term, we are effectively ignoring the effects of shock heating.
    - Consequently, we might want to overwrite the "thermal energy" to capture the effects of shock heating. 
    - The precise condition that dictates when we overwrite the "thermal energy" field involves a comparison of {math}`\eta_2` and the values in neighboring cells. When {math}`\eta_2` is too high, we would effectively exclude shock-heating from weaker shocks. When {math}`\eta_2` is too low we may include spurious heating that is introduced by the truncation error of {math}`(E-E_{\rm kinetic})`.
    - **NOTE:** [Bryan+ 2014](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...19B) call this step "synchronization" - we find that name somewhat confusing since it may imply a bidirectional update (updating both "thermal energy" and {math}`E`).

In practice, Cholla does something slightly different:
1. It implements step 1 exactly as described above.
2. Step 2 is pretty much the same. However, we also overwrite the "thermal energy" field when the quotient of "thermal energy" divided by {math}`E` exceeds {math}`\eta_1`.
3. We add an additional step after step 2, where we overwrite {math}`E` with the sum of the "thermal energy" field and {math}`E_{\rm kinetic}`.
   - This is useful for book-keeping reasons throughout the rest of the codebase.
   - Something like this step is commonly implemented in other simulation codes (for example, Enzo effectively does this in any configuration involving radiative cooling physics).

## Configuring the Dual Energy Formalism

To use the dual-energy formalism, you just need to define the `DE` macro at compile-time

The `DE_ETA_1` and `DE_ETA_2` macros are automatically defined within Cholla. At the time of writing this page, Cholla sets `DE_ETA_1` to different values based on whether it is configured in cosmology. In cosmological simulations, `DE_ETA_1` is always set to 1. This means that the separately advected "thermal energy" field is **always** prioritized over {math}`E` when computing the thermal pressure.

At this time, the dual-energy formalism is **NOT** compatible with MHD.