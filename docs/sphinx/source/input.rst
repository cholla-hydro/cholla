Input files
===========
This page briefly describes parameters that may be defined in the input file. All parameters are case sensitive. Required parameters for all setups are listed first, followed by optional parameters required for specific Makefile flags or initial conditions. A `sample <https://github.com/cholla-hydro/cholla/blob/master/tests/sample.txt>`_ parameter file can be found in the `tests <https://github.com/cholla-hydro/cholla/tree/master/tests>`_ directory.

Required parameters
-------------------
For 1D problems, ny and nz should be set to 1. For 2D problems, nz should be set to 1.

+------------------+-------------------------------+
|     Parameter    |          Description          |
+==================+===============================+
|    nx, ny, nz    |      Number of grid cells     |
+------------------+-------------------------------+
| xmin, ymin, zmin |    Lower domain boundaries    |
+------------------+-------------------------------+
| xlen, ylen, zlen |      Global domain length     |
+------------------+-------------------------------+
|       tout       |       Final output time       |
+------------------+-------------------------------+
|      outstep     | Code time interval for output |
+------------------+-------------------------------+
|       gamma      |    Ratio of specific heats    |
+------------------+-------------------------------+
|       init       |   Name of initial conditions  |
+------------------+-------------------------------+
|      outdir      |    Path to output directory   |
+------------------+-------------------------------+

All dimensionful parameters are in code units. Some further notes:

- gamma: Cholla currently supports only a single gamma for all cells.
- init: Case sensitive. Current options include Constant, Sound_Wave, Square_Wave, Riemann, Shu_Osher, Blast_1D, KH, KH_res_ind, Rayleigh_Taylor, Implosion_2D, Gresho, Noh_2D, Noh_3D, Disk_2D, Disk_3D, Spherical_Overpressure_3D, Spherical_Overdensity_3D, Uniform, Zeldovich_Pancake, and Read_Grid. See `initial_conditions.cpp <https://github.com/cholla-hydro/cholla/blob/master/src/initial_conditions.cpp>`_ for more information about each option. Sample input parameter files for many of these problems can be found in the `tests <https://github.com/cholla-hydro/cholla/tree/master/tests>`_ directory.
- outstep: If more than one type of output file is defined, whis will correspond to the most frequent output.

Boundary conditions
...................
Given as a number, options include 1 (periodic), 2 (reflective), 3 (transmissive), 4 (custom). Can use different boundary conditions for each boundary.

+---------------------------------+-------------------------------+
|             Parameter           |          Description          |
+=================================+===============================+
|    xl_bcnd, yl_bcnd, zl_bcnd    |   Lower boundary conditions   |
+---------------------------------+-------------------------------+
|    xu_bcnd, yu_bcnd, zu_bcnd    |   Upper boundary conditions   |
+---------------------------------+-------------------------------+
