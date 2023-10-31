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

from cat_dset_3D import copy_header, common_cli

# ==============================================================================
def concat_slice(source_directory: pathlib.Path,
                 output_directory: pathlib.Path,
                 num_processes: int,
                 output_number: int,
                 concat_xy: bool = True,
                 concat_yz: bool = True,
                 concat_xz: bool = True,
                 skip_fields: list = [],
                 destination_dtype: np.dtype = None,
                 compression_type: str = None,
                 compression_options: str = None,
                 chunking = None):
  """Concatenate slice HDF5 Cholla datasets. i.e. take the single files
  generated per process and concatenate them into a single, large file. This
  function concatenates a single output time and can be called multiple times,
  potentially in parallel, to concatenate multiple output times.

  Args:
      source_directory (pathlib.Path): The directory containing the unconcatenated files
      output_directory (pathlib.Path): The directory containing the new concatenated files
      num_processes (int): The number of ranks that Cholla was run with
      output_number (int): The output number to concatenate
      concat_xy (bool, optional): If True then concatenate the XY slice. Defaults to True.
      concat_yz (bool, optional): If True then concatenate the YZ slice. Defaults to True.
      concat_xz (bool, optional): If True then concatenate the XZ slice. Defaults to True.
      skip_fields (list, optional): List of fields to skip concatenating. Defaults to [].
      destination_dtype (np.dtype, optional): The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.
      compression_type (str, optional): What kind of compression to use on the output data. Defaults to None.
      compression_options (str, optional): What compression settings to use if compressing. Defaults to None.
      chunking (bool or tuple, optional): Whether or not to use chunking and the chunk size. Defaults to None.
  """

  # Error checking
  assert num_processes > 1, 'num_processes must be greater than 1'
  assert output_number >= 0, 'output_number must be greater than or equal to 0'

  # Open destination file and first file for getting metadata
  destination_file = h5py.File(output_directory / f'{output_number}_slice.h5', 'w-')

  # Setup the output file
  with h5py.File(source_directory / f'{output_number}_slice.h5.0', 'r') as source_file:
    # Copy over header
    destination_file = copy_header(source_file, destination_file)

    # Get a list of all datasets in the source file
    datasets_to_copy = list(source_file.keys())

    # Filter the datasets to only include those that need to be copied
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

      slice_shape = __get_slice_shape(source_file, dataset)

      destination_file.create_dataset(name=dataset,
                                      shape=slice_shape,
                                      dtype=dtype,
                                      chunks=chunking,
                                      compression=compression_type,
                                      compression_opts=compression_options)

  # Copy data
  for rank in range(num_processes):
    # Open source file
    source_file = h5py.File(source_directory / f'{output_number}_slice.h5.{rank}', 'r')

    # Loop through and copy datasets
    for dataset in datasets_to_copy:
      # Determine locations and shifts for writing
      (i0_start, i0_end, i1_start, i1_end), file_in_slice = __write_bounds_slice(source_file, dataset)

      if file_in_slice:
        # Copy the data
        destination_file[dataset][i0_start:i0_end,
                                  i1_start:i1_end] = source_file[dataset]

    # Now that the copy is done we close the source file
    source_file.close()

  # Close destination file now that it is fully constructed
  destination_file.close()
# ==============================================================================

# ==============================================================================
def __get_slice_shape(source_file: h5py.File, dataset: str):
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
def __write_bounds_slice(source_file: h5py.File, dataset: str):
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

  cli = common_cli()
  cli.add_argument('--disable-xy', default=True, action='store_false', help='Disables concating the XY slice.')
  cli.add_argument('--disable-yz', default=True, action='store_false', help='Disables concating the YZ slice.')
  cli.add_argument('--disable-xz', default=True, action='store_false', help='Disables concating the XZ slice.')
  args = cli.parse_args()

  # Perform the concatenation
  for output in args.concat_outputs:
    concat_slice(source_directory=args.source_directory,
                 output_directory=args.output_directory,
                 num_processes=args.num_processes,
                 output_number=output,
                 concat_xy=args.disable_xy,
                 concat_yz=args.disable_yz,
                 concat_xz=args.disable_xz,
                 skip_fields=args.skip_fields,
                 destination_dtype=args.dtype,
                 compression_type=args.compression_type,
                 compression_options=args.compression_opts,
                 chunking=args.chunking)

  print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
