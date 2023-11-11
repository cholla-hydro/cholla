#!/usr/bin/env python3
"""
Python script for concatenating 3D hdf5 datasets. Includes a CLI for concatenating Cholla HDF5 datasets and can be
imported into other scripts where the `concat_3d_field` function can be used to concatenate the datasets.

Generally the easiest way to import this script is to add the `python_scripts` directory to your python path in your
script like this:
```
import sys
sys.path.append('/PATH/TO/CHOLLA/python_scripts')
import concat_3d_data
```
"""

import h5py
import numpy as np
import argparse
import pathlib

# ======================================================================================================================
def concat_3d_output(source_directory: pathlib.Path,
                     output_directory: pathlib.Path,
                     num_processes: int,
                     output_number: int,
                     skip_fields: list = [],
                     destination_dtype: np.dtype = None,
                     compression_type: str = None,
                     compression_options: str = None,
                     chunking = None):
  """Concatenate a single 3D HDF5 Cholla dataset. i.e. take the single files generated per process and concatenate them into a
  single, large file.

  Args:
      source_directory (pathlib.Path): The directory containing the unconcatenated files
      output_directory (pathlib.Path): The directory containing the new concatenated files
      num_processes (int): The number of ranks that Cholla was run with
      output_number (int): The output number to concatenate
      skip_fields (list, optional): List of fields to skip concatenating. Defaults to [].
      destination_dtype (np.dtype, optional): The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.
      compression_type (str, optional): What kind of compression to use on the output data. Defaults to None.
      compression_options (str, optional): What compression settings to use if compressing. Defaults to None.
      chunking (bool or tuple, optional): Whether or not to use chunking and the chunk size. Defaults to None.
  """

  # Error checking
  assert num_processes > 1, 'num_processes must be greater than 1'
  assert output_number >= 0, 'output_number must be greater than or equal to 0'

  # open the output file for writing (fail if it exists)
  destination_file = h5py.File(output_directory / f'{output_number}.h5', 'w-')

  # Setup the output file
  with h5py.File(source_directory / f'{output_number}.h5.0', 'r') as source_file:
    # Copy header data
    destination_file = copy_header(source_file, destination_file)

    # Create the datasets in the output file
    datasets_to_copy = list(source_file.keys())
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not dataset in skip_fields]

    for dataset in datasets_to_copy:
      dtype = source_file[dataset].dtype if (destination_dtype == None) else destination_dtype

      data_shape = source_file.attrs['dims']

      if dataset == 'magnetic_x': data_shape[0] += 1
      if dataset == 'magnetic_y': data_shape[1] += 1
      if dataset == 'magnetic_z': data_shape[2] += 1

      destination_file.create_dataset(name=dataset,
                                      shape=data_shape,
                                      dtype=dtype,
                                      chunks=chunking,
                                      compression=compression_type,
                                      compression_opts=compression_options)

  # loop over files for a given output
  for i in range(0, num_processes):
    # open the input file for reading
    source_file = h5py.File(source_directory / f'{output_number}.h5.{i}', 'r')

    # Compute the offset slicing
    nx_local, ny_local, nz_local = source_file.attrs['dims_local']
    x_start, y_start, z_start    = source_file.attrs['offset']
    x_end, y_end, z_end          = x_start+nx_local, y_start+ny_local, z_start+nz_local

    # write data from individual processor file to correct location in concatenated file
    for dataset in datasets_to_copy:
      magnetic_offset = [0,0,0]
      if dataset == 'magnetic_x': magnetic_offset[0] = 1
      if dataset == 'magnetic_y': magnetic_offset[1] = 1
      if dataset == 'magnetic_z': magnetic_offset[2] = 1

      destination_file[dataset][x_start:x_end+magnetic_offset[0],
                                y_start:y_end+magnetic_offset[1],
                                z_start:z_end+magnetic_offset[2]] = source_file[dataset]

    # Now that the copy is done we close the source file
    source_file.close()

  # Close destination file now that it is fully constructed
  destination_file.close()
# ======================================================================================================================

# ==============================================================================
def copy_header(source_file: h5py.File, destination_file: h5py.File):
  """Copy the attributes of one HDF5 file to another, skipping all fields that are specific to an individual rank

  Args:
      source_file (h5py.File): The source file
      destination_file (h5py.File): The destination file

  Returns:
      h5py.File: The destination file with the new header attributes
  """
  fields_to_skip = ['dims_local', 'offset']

  for attr_key in source_file.attrs.keys():
    if attr_key not in fields_to_skip:
      destination_file.attrs[attr_key] = source_file.attrs[attr_key]

  return destination_file
# ==============================================================================

# ==============================================================================
def common_cli() -> argparse.ArgumentParser:
  """This function provides the basis for the common CLI amongst the various concatenation scripts. It returns an
  `argparse.ArgumentParser` object to which additional arguments can be passed before the final `.parse_args()` method
  is used.
  """

  # ============================================================================
  # Function used to parse the `--concat-output` argument
  def concat_output(raw_argument: str) -> list:
    # Check if the string is empty
    if len(raw_argument) < 1:
      raise ValueError('The --concat-output argument must not be of length zero.')

    # Strip unneeded characters
    cleaned_argument = raw_argument.replace(' ', '')
    cleaned_argument = cleaned_argument.replace('[', '')
    cleaned_argument = cleaned_argument.replace(']', '')

    # Check that it only has the allowed characters
    allowed_charaters = set('0123456789,-')
    if not set(cleaned_argument).issubset(allowed_charaters):
      raise ValueError("Argument contains incorrect characters. Should only contain '0-9', ',', and '-'.")

    # Split on commas
    cleaned_argument = cleaned_argument.split(',')

    # Generate the final list
    iterable_argument = set()
    for arg in cleaned_argument:
      if '-' not in arg:
        if int(arg) < 0:
          raise ValueError()
        iterable_argument.add(int(arg))
      else:
        start, end = arg.split('-')
        start, end = int(start), int(end)
        if end < start:
          raise ValueError('The end of a range must be larger than the start of the range.')
        if start < 0:
          raise ValueError()
        iterable_argument = iterable_argument.union(set(range(start, end+1)))

    return iterable_argument
  # ============================================================================

  # ============================================================================
  def positive_int(raw_argument: str) -> int:
    arg = int(raw_argument)
    if arg < 0:
      raise ValueError('Argument must be 0 or greater.')

    return arg
  # ============================================================================

  # ============================================================================
  def skip_fields(raw_argument: str) -> list:
    # Strip unneeded characters
    cleaned_argument = raw_argument.replace(' ', '')
    cleaned_argument = cleaned_argument.replace('[', '')
    cleaned_argument = cleaned_argument.replace(']', '')
    cleaned_argument = cleaned_argument.split(',')

    return cleaned_argument
  # ============================================================================

  # ============================================================================
  def chunk_arg(raw_argument: str):
    # Strip unneeded characters
    cleaned_argument = raw_argument.replace(' ', '')
    cleaned_argument = cleaned_argument.replace('(', '')
    cleaned_argument = cleaned_argument.replace(')', '')

    # Check that it only has the allowed characters
    allowed_charaters = set('0123456789,')
    if not set(cleaned_argument).issubset(allowed_charaters):
      raise ValueError("Argument contains incorrect characters. Should only contain '0-9', ',', and '-'.")

    # Convert to a tuple and return
    return tuple([int(i) for i in cleaned_argument.split(',')])
  # ============================================================================

  # Initialize the CLI
  cli = argparse.ArgumentParser()

  # Required Arguments
  cli.add_argument('-s', '--source-directory', type=pathlib.Path,  required=True, help='The path to the directory for the source HDF5 files.')
  cli.add_argument('-o', '--output-directory', type=pathlib.Path,  required=True, help='The path to the directory to write out the concatenated HDF5 files.')
  cli.add_argument('-n', '--num-processes',    type=positive_int,  required=True, help='The number of processes that were used')
  cli.add_argument('-c', '--concat-outputs',   type=concat_output, required=True, help='Which outputs to concatenate. Can be a single number (e.g. 8), a range (e.g. 2-9), or a list (e.g. [1,2,3]). Ranges are inclusive')

  # Optional Arguments
  cli.add_argument('--skip-fields',            type=skip_fields,   default=[],   help='List of fields to skip concatenating. Defaults to empty.')
  cli.add_argument('--dtype',                  type=str,           default=None, help='The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.')
  cli.add_argument('--compression-type',       type=str,           default=None, help='What kind of compression to use on the output data. Defaults to None.')
  cli.add_argument('--compression-opts',       type=str,           default=None, help='What compression settings to use if compressing. Defaults to None.')
  cli.add_argument('--chunking',               type=chunk_arg,     default=None, nargs='?', const=True, help='Enable chunking of the output file. Default is `False`. If set without an argument then the chunk size will be automatically chosen or a tuple can be passed to indicate the chunk size desired.')

  return cli
# ==============================================================================

if __name__ == '__main__':
  from timeit import default_timer
  start = default_timer()

  cli = common_cli()
  args = cli.parse_args()

  # Perform the concatenation
  for output in args.concat_outputs:
    concat_3d_output(source_directory=args.source_directory,
                     output_directory=args.output_directory,
                     num_processes=args.num_processes,
                     output_number=output,
                     skip_fields=args.skip_fields,
                     destination_dtype=args.dtype,
                     compression_type=args.compression_type,
                     compression_options=args.compression_opts,
                     chunking=args.chunking)

  print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
