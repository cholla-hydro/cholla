Paris and Henry
====
This directory contains classes for the Paris Poisson solvers and Henry 3D FFT filter.

*ParisPeriodic*
----
A 3D Poisson solver that expects periodic boundary conditions.

*ParisPeriodic* calls the FFT filter provided by the *HenryPeriodic* class, where it provides a C++ lambda function that solves the Poisson equation in frequency space.
It assumes fields in a 3D block distribution with no ghost cells.
It is used by the Cholla class *PotentialParis3D* to solve Poisson problems with periodic boundary conditions.

To use:
- Construct a *ParisPeriodic* object using information about the global domain and local MPI task.
- Use the method *ParisPeriodic::bytes()* to get the array size needed for the solver arguments.
- Allocate input *density* and output *potential* arrays of at least the required size.
- Call *ParisPeriodic::solve()* to compute a *potential* from an input *density*.

See the comments in `ParisPeriodic.hpp` for details on the methods and their arguments.

*HenryPeriodic*
----
A generic distributed 3D FFT filter class.

*HenryPeriodic* performs a 3D FFT, applies a frequency-space function that is provided as an argument, and applies an inverse 3D FFT.
It assumes fields in a 3D block distribution with no ghost cells.
It is used by *ParisPeriodic* to solve periodic Poisson problems, and it is intended for more-general use for other FFT-filtering operations.

To use outside of *ParisPeriodic*:
- Construct a *HenryPeriodic* object using information about the global domain and local MPI rank.
- Use the method *HenryPeriodic::bytes()* to get the array size needed for the filter arguments.
- Allocate input *before* and output *after* arrays of at least the required size.
- Call *HenryPeriodic::filter()* with these arrays, along with a functor or lambda function that performs the desired frequency-space filter.
The function should take arguments specifying a 3D coordinate in frequency space, along with an input complex value.
It should return a filtered complex value.

See the comments in `HenryPeriodic.hpp` for details on the methods, their arguments, and the expected prototype of the filter function.

See the implementation of *ParisPeriodic::solve()* in `ParisPeriodic.cpp` for an example of using *HenryPeriodic::filter()*.

*PoissonZero3DBlockedGPU*
----
A 3D Poisson solver that expects zero-valued boundary conditions.

*PoissonZero3DBlockedGPU* uses discrete sine transforms (DSTs) instead of Fourier transforms to enforce zero-valued, non-periodic boundary conditions.
It is currently a monolithic class, not depenedent on a *Henry* class.
It is used by the Cholla class *PotentialParisGalactic* to solve Poisson problems with non-zero, non-periodic, analytic boundary conditions.

*PotentialParisGalactic::Get_Potential()* uses *PoissonZero3DBlockedGPU::solve()* as follows.
- Subtract an analytic density from the input density, where the analytic density matches the input density at the domain boundaries.
This results in a density with zero-valued boundaries.
- Call *PoissonZero3DBlockedGPU::solve()* with this density with zero-valued boundaries.
- Add an analytic potential to the resulting potential, where the analytic potential is the solution to the Poisson equation for the analytic density that was subtracted from the input density.
The resulting sum of potentials is the solution to the Poisson problem for the full input density.

