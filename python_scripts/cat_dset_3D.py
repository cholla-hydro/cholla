#!/usr/bin/env python3
"""
Python script for concatenating 3D hdf5 datasets. Includes a CLI for concatenating Cholla HDF5 datasets and can be
imported into other scripts where the `concat_3d` function can be used to concatenate the datasets.

Generally the easiest way to import this script is to add the `python_scripts` directory to your python path in your
script like this:
```
import sys
sys.path.append('/PATH/TO/CHOLLA/python_scripts')
import cat_dset_3D
```
"""

import h5py
import numpy as np
import argparse
import pathlib

def main():
  """This function handles the CLI argument parsing and is only intended to be used when this script is invoked from the
  command line. If you're importing this file then use the `concat_3d` function directly.
  """
  # Argument handling
  cli = argparse.ArgumentParser()
  # Required Arguments
  cli.add_argument('-s', '--start_num',     type=int, required=True, help='The first output step to concatenate')
  cli.add_argument('-e', '--end_num',       type=int, required=True, help='The last output step to concatenate')
  cli.add_argument('-n', '--num_processes', type=int, required=True, help='The number of processes that were used')
  # Optional Arguments
  cli.add_argument('-i', '--input_dir',  type=pathlib.Path, default=pathlib.Path.cwd(), help='The input directory.')
  cli.add_argument('-o', '--output_dir', type=pathlib.Path, default=pathlib.Path.cwd(), help='The output directory.')
  args = cli.parse_args()

  # Perform the concatenation
  concat_3d(start_num=args.start_num,
            end_num=args.end_num,
            num_processes=args.num_processes,
            input_dir=args.input_dir,
            output_dir=args.output_dir)


# ======================================================================================================================
def concat_3d(start_num: int,
              end_num: int,
              num_processes: int,
              input_dir: pathlib.Path = pathlib.Path.cwd(),
              output_dir: pathlib.Path = pathlib.Path.cwd()):
  """Concatenate 3D HDF5 Cholla datasets. i.e. take the single files generated per process and concatenate them into a
  single, large file. All outputs from start_num to end_num will be concatenated.

  Args:
      start_num (int): The first output step to concatenate
      end_num (int): The last output step to concatenate
      num_processes (int): The number of processes that were used
      input_dir (pathlib.Path, optional): The input directory. Defaults to pathlib.Path.cwd().
      output_dir (pathlib.Path, optional): The output directory. Defaults to pathlib.Path.cwd().
  """

  # Error checking
  assert start_num >= 0, 'start_num must be greater than or equal to 0'
  assert end_num >= 0, 'end_num must be greater than or equal to 0'
  assert start_num <= end_num, 'end_num should be greater than or equal to start_num'
  assert num_processes > 1, 'num_processes must be greater than 1'

  # loop over outputs
  for n in range(start_num, end_num+1):

    # loop over files for a given output
    for i in range(0, num_processes):

      # open the output file for writing (don't overwrite if exists)
      fileout = h5py.File(output_dir / f'{n}.h5', 'a')
      # open the input file for reading
      filein = h5py.File(input_dir / f'{n}.h5.{i}', 'r')
      # read in the header data from the input file
      head = filein.attrs

      # if it's the first input file, write the header attributes
      # and create the datasets in the output file
      if (i == 0):
        nx = head['dims'][0]
        ny = head['dims'][1]
        nz = head['dims'][2]
        fileout.attrs['dims'] = [nx, ny, nz]
        fileout.attrs['gamma'] = [head['gamma'][0]]
        fileout.attrs['t'] = [head['t'][0]]
        fileout.attrs['dt'] = [head['dt'][0]]
        fileout.attrs['n_step'] = [head['n_step'][0]]

        units = ['time_unit', 'mass_unit', 'length_unit', 'energy_unit', 'velocity_unit', 'density_unit']
        for unit in units:
          fileout.attrs[unit] = [head[unit][0]]

        d  = fileout.create_dataset("density", (nx, ny, nz), chunks=True, dtype=filein['density'].dtype)
        mx = fileout.create_dataset("momentum_x", (nx, ny, nz), chunks=True, dtype=filein['momentum_x'].dtype)
        my = fileout.create_dataset("momentum_y", (nx, ny, nz), chunks=True, dtype=filein['momentum_y'].dtype)
        mz = fileout.create_dataset("momentum_z", (nx, ny, nz), chunks=True, dtype=filein['momentum_z'].dtype)
        E  = fileout.create_dataset("Energy", (nx, ny, nz), chunks=True, dtype=filein['Energy'].dtype)
        try:
          GE = fileout.create_dataset("GasEnergy", (nx, ny, nz), chunks=True, dtype=filein['GasEnergy'].dtype)
        except KeyError:
          print('No Dual energy data present');
        try:
          bx = fileout.create_dataset("magnetic_x", (nx+1, ny, nz), chunks=True, dtype=filein['magnetic_x'].dtype)
          by = fileout.create_dataset("magnetic_y", (nx, ny+1, nz), chunks=True, dtype=filein['magnetic_y'].dtype)
          bz = fileout.create_dataset("magnetic_z", (nx, ny, nz+1), chunks=True, dtype=filein['magnetic_z'].dtype)
        except KeyError:
          print('No magnetic field data present');

      # write data from individual processor file to
      # correct location in concatenated file
      nxl = head['dims_local'][0]
      nyl = head['dims_local'][1]
      nzl = head['dims_local'][2]
      xs = head['offset'][0]
      ys = head['offset'][1]
      zs = head['offset'][2]
      fileout['density'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = filein['density']
      fileout['momentum_x'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['momentum_x']
      fileout['momentum_y'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['momentum_y']
      fileout['momentum_z'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['momentum_z']
      fileout['Energy'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl]  = filein['Energy']
      try:
        fileout['GasEnergy'][xs:xs+nxl,ys:ys+nyl,zs:zs+nzl] = filein['GasEnergy']
      except KeyError:
          print('No Dual energy data present');
      try:
        fileout['magnetic_x'][xs:xs+nxl+1, ys:ys+nyl,   zs:zs+nzl]   = filein['magnetic_x']
        fileout['magnetic_y'][xs:xs+nxl,   ys:ys+nyl+1, zs:zs+nzl]   = filein['magnetic_y']
        fileout['magnetic_z'][xs:xs+nxl,   ys:ys+nyl,   zs:zs+nzl+1] = filein['magnetic_z']
      except KeyError:
          print('No magnetic field data present');

      filein.close()

    fileout.close()
# ======================================================================================================================

if __name__ == '__main__':
  main()
