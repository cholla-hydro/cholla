#!/usr/bin/env python3
"""
Python script for concatenating slice hdf5 datasets for when -DSLICES is turned
on in Cholla. Includes a CLI for concatenating Cholla HDF5 datasets and can be
imported into other scripts where the `concat_slice` function can be used to
concatenate the HDF5 files.

Generally the easiest way to import this script is to add the `python_scripts`
directory to your python path in your script like this:
```
import sys
sys.path.append('/PATH/TO/CHOLLA/python_scripts')
import cat_slice
```
"""

import h5py
import argparse
import pathlib
import numpy as np

from cat_dset_3D import copy_header

# ==============================================================================
def main():
  """This function handles the CLI argument parsing and is only intended to be used when this script is invoked from the
  command line. If you're importing this file then use the `concat_slice` function directly.
  """
  # Argument handling
  cli = argparse.ArgumentParser()
  # Required Arguments
  cli.add_argument('-s', '--source-directory', type=pathlib.Path, required=True, help='The path to the source HDF5 files.')
  cli.add_argument('-o', '--output-file',      type=pathlib.Path, required=True, help='The path and filename of the concatenated file.')
  cli.add_argument('-n', '--num-processes',    type=int,          required=True, help='The number of processes that were used to generate the slices.')
  cli.add_argument('-t', '--output-num',     type=int,          required=True, help='The output number to be concatenated')
  # Optional Arguments
  cli.add_argument('--xy',               type=bool, default=True, help='If True then concatenate the XY slice. Defaults to True.')
  cli.add_argument('--yz',               type=bool, default=True, help='If True then concatenate the YZ slice. Defaults to True.')
  cli.add_argument('--xz',               type=bool, default=True, help='If True then concatenate the XZ slice. Defaults to True.')
  cli.add_argument('--skip-fields',      type=list, default=[],   help='List of fields to skip concatenating. Defaults to empty.')
  cli.add_argument('--dtype',            type=str,  default=None, help='The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.')
  cli.add_argument('--compression-type', type=str,  default=None, help='What kind of compression to use on the output data. Defaults to None.')
  cli.add_argument('--compression-opts', type=str,  default=None, help='What compression settings to use if compressing. Defaults to None.')
  args = cli.parse_args()

  # Perform the concatenation
  concat_slice(source_directory=args.source_directory,
               destination_file_path=args.output_file,
               num_ranks=args.num_processses,
               output_number=args.output_num,
               concat_xy=args.xy,
               concat_yz=args.yz,
               concat_xz=args.xz,
               skip_fields=args.skip_fields,
               destination_dtype=args.dtype,
               compression_type=args.compression_type,
               compression_options=args.compression_opts)
# ==============================================================================

# ==============================================================================
def concat_slice(source_directory: pathlib.Path,
                 destination_file_path: pathlib.Path,
                 num_ranks: int,
                 output_number: int,
                 concat_xy: bool = True,
                 concat_yz: bool = True,
                 concat_xz: bool = True,
                 skip_fields: list = [],
                 destination_dtype: np.dtype = None,
                 compression_type: str = None,
                 compression_options: str = None):
  """Concatenate slice HDF5 Cholla datasets. i.e. take the single files
  generated per process and concatenate them into a single, large file. This
  function concatenates a single output time and can be called multiple times,
  potentially in parallel, to concatenate multiple output times.

  Args:
      source_directory (pathlib.Path): The directory containing the unconcatenated files
      destination_file_path (pathlib.Path): The path and name of the new concatenated file
      num_ranks (int): The number of ranks that Cholla was run with
      output_number (int): The output number to concatenate
      concat_xy (bool, optional): If True then concatenate the XY slice. Defaults to True.
      concat_yz (bool, optional): If True then concatenate the YZ slice. Defaults to True.
      concat_xz (bool, optional): If True then concatenate the XZ slice. Defaults to True.
      skip_fields (list, optional): List of fields to skip concatenating. Defaults to [].
      destination_dtype (np.dtype, optional): The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.
      compression_type (str, optional): What kind of compression to use on the output data. Defaults to None.
      compression_options (str, optional): What compression settings to use if compressing. Defaults to None.
  """
  # Open destination file and first file for getting metadata
  source_file = h5py.File(source_directory / f'{output_number}_slice.h5.0', 'r')
  destination_file = h5py.File(destination_file_path, 'w')

  # Copy over header
  destination_file = copy_header(source_file, destination_file)

  # Get a list of all datasets in the source file
  datasets_to_copy = list(source_file.keys())

  # Filter the datasets to only include those I wish to copy
  if not concat_xy:
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not 'xy' in dataset]
  if not concat_yz:
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not 'yz' in dataset]
  if not concat_xz:
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not 'xz' in dataset]
  datasets_to_copy = [dataset for dataset in datasets_to_copy if not dataset in skip_fields]

  # Create the datasets in the destination file
  for dataset in datasets_to_copy:
    dtype = source_file[dataset].dtype if (destination_dtype == None) else destination_dtype

    slice_shape = get_slice_shape(source_file, dataset)

    destination_file.create_dataset(name=dataset,
                                    shape=slice_shape,
                                    dtype=dtype,
                                    compression=compression_type,
                                    compression_opts=compression_options)

  # Close source file in prep for looping through source files
  source_file.close()

  # Copy data
  for rank in range(num_ranks):
    # Open source file
    source_file = h5py.File(source_directory / f'{output_number}_slice.h5.{rank}', 'r')

    # Loop through and copy datasets
    for dataset in datasets_to_copy:
      # Determine locations and shifts for writing
      (i0_start, i0_end, i1_start, i1_end), file_in_slice = write_bounds(source_file, dataset)

      if file_in_slice:
        # Copy the data
        destination_file[dataset][i0_start:i0_end, i1_start:i1_end] = source_file[dataset]

    # Now that the copy is done we close the source file
    source_file.close()

  # Close destination file now that it is fully constructed
  destination_file.close()
# ==============================================================================

# ==============================================================================
def get_slice_shape(source_file: h5py.File, dataset: str):
  """Determine the shape of the full slice in a dataset

  Args:
      source_file (h5py.File): The source file the get the shape information from
      dataset (str): The dataset to get the shape of

  Raises:
      ValueError: If the dataset name isn't a slice name

  Returns:
      tuple: The 2D dimensions of the slice
  """
  nx, ny, nz = source_file.attrs['dims']

  if 'xy' in dataset:
    slice_dimensions = (nx, ny)
  elif 'yz' in dataset:
    slice_dimensions = (ny, nz)
  elif 'xz' in dataset:
    slice_dimensions = (nx, nz)
  else:
    raise ValueError(f'Dataset "{dataset}" is not a slice.')

  return slice_dimensions
# ==============================================================================

# ==============================================================================
def write_bounds(source_file: h5py.File, dataset: str):
  """Determine the bounds of the concatenated file to write to

  Args:
      source_file (h5py.File): The source file to read from
      dataset (str): The name of the dataset to read from the source file

  Raises:
      ValueError: If the dataset name isn't a slice name

  Returns:
      tuple: The write bounds for the concatenated file to be used like `output_file[dataset][return[0]:return[1], return[2]:return[3]]
  """
  nx, ny, nz                   = source_file.attrs['dims']
  nx_local, ny_local, nz_local = source_file.attrs['dims_local']
  x_start, y_start, z_start    = source_file.attrs['offset']

  if 'xy' in dataset:
    file_in_slice = z_start <= nz//2 <= z_start+nz_local
    bounds = (x_start, x_start+nx_local, y_start, y_start+ny_local)
  elif 'yz' in dataset:
    file_in_slice = x_start <= nx//2 <= x_start+nx_local
    bounds = (y_start, y_start+ny_local, z_start, z_start+nz_local)
  elif 'xz' in dataset:
    file_in_slice = y_start <= ny//2 <= y_start+ny_local
    bounds = (x_start, x_start+nx_local, z_start, z_start+nz_local)
  else:
    raise ValueError(f'Dataset "{dataset}" is not a slice.')

  return bounds, file_in_slice
# ==============================================================================

if __name__ == '__main__':
  from timeit import default_timer
  start = default_timer()
  main()
  print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
