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

# ======================================================================================================================
def main():
  """This function handles the CLI argument parsing and is only intended to be used when this script is invoked from the
  command line. If you're importing this file then use the `concat_3d` or `concat_3d_single` functions directly.
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
  cli.add_argument('--skip-fields',      type=list, default=[],   help='List of fields to skip concatenating. Defaults to empty.')
  cli.add_argument('--dtype',            type=str,  default=None, help='The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.')
  cli.add_argument('--compression-type', type=str,  default=None, help='What kind of compression to use on the output data. Defaults to None.')
  cli.add_argument('--compression-opts', type=str,  default=None, help='What compression settings to use if compressing. Defaults to None.')
  args = cli.parse_args()

  # Perform the concatenation
  concat_3d(start_num=args.start_num,
            end_num=args.end_num,
            num_processes=args.num_processes,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            skip_fields=args.skip_fields,
            destination_dtype=args.dtype,
            compression_type=args.compression_type,
            compression_options=args.compression_opts)
# ======================================================================================================================

# ======================================================================================================================
def concat_3d(start_num: int,
              end_num: int,
              num_processes: int,
              input_dir: pathlib.Path = pathlib.Path.cwd(),
              output_dir: pathlib.Path = pathlib.Path.cwd(),
              skip_fields: list = [],
              destination_dtype: np.dtype = None,
              compression_type: str = None,
              compression_options: str = None):
  """Concatenate 3D HDF5 Cholla datasets. i.e. take the single files generated per process and concatenate them into a
  single, large file. All outputs from start_num to end_num will be concatenated.

  Args:
      start_num (int): The first output step to concatenate
      end_num (int): The last output step to concatenate
      num_processes (int): The number of processes that were used
      input_dir (pathlib.Path, optional): The input directory. Defaults to pathlib.Path.cwd().
      output_dir (pathlib.Path, optional): The output directory. Defaults to pathlib.Path.cwd().
      skip_fields (list, optional): List of fields to skip concatenating. Defaults to [].
      destination_dtype (np.dtype, optional): The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.
      compression_type (str, optional): What kind of compression to use on the output data. Defaults to None.
      compression_options (str, optional): What compression settings to use if compressing. Defaults to None.
  """

  # Error checking
  assert start_num >= 0, 'start_num must be greater than or equal to 0'
  assert end_num >= 0, 'end_num must be greater than or equal to 0'
  assert start_num <= end_num, 'end_num should be greater than or equal to start_num'
  assert num_processes > 1, 'num_processes must be greater than 1'

  # loop over outputs
  for n in range(start_num, end_num+1):
    concat_3d_single(output_number=n,
                     num_processes=num_processes,
                     input_dir=input_dir,
                     output_dir=output_dir,
                     skip_fields=skip_fields,
                     destination_dtype=destination_dtype,
                     compression_type=compression_type,
                     compression_options=compression_options)
# ======================================================================================================================

# ======================================================================================================================
def concat_3d_single(output_number: int,
                     num_processes: int,
                     input_dir: pathlib.Path = pathlib.Path.cwd(),
                     output_dir: pathlib.Path = pathlib.Path.cwd(),
                     skip_fields: list = [],
                     destination_dtype: np.dtype = None,
                     compression_type: str = None,
                     compression_options: str = None):
  """Concatenate a single 3D HDF5 Cholla dataset. i.e. take the single files generated per process and concatenate them into a
  single, large file.

  Args:
      output_number (int): The output
      end_num (int): The last output step to concatenate
      num_processes (int): The number of processes that were used
      input_dir (pathlib.Path, optional): The input directory. Defaults to pathlib.Path.cwd().
      output_dir (pathlib.Path, optional): The output directory. Defaults to pathlib.Path.cwd().
      skip_fields (list, optional): List of fields to skip concatenating. Defaults to [].
      destination_dtype (np.dtype, optional): The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.
      compression_type (str, optional): What kind of compression to use on the output data. Defaults to None.
      compression_options (str, optional): What compression settings to use if compressing. Defaults to None.
  """

  # Error checking
  assert num_processes > 1, 'num_processes must be greater than 1'
  assert output_number >= 0, 'output_number must be greater than or equal to 0'

  # open the output file for writing (don't overwrite if exists)
  fileout = h5py.File(output_dir / f'{output_number}.h5', 'a')

  # Setup the output file
  with h5py.File(input_dir / f'{output_number}.h5.0', 'r') as source_file:
    # Copy header data
    fileout = copy_header(source_file, fileout)

    # Create the datasets in the output file
    datasets_to_copy = list(source_file.keys())
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not dataset in skip_fields]

    for dataset in datasets_to_copy:
      dtype = source_file[dataset].dtype if (destination_dtype == None) else destination_dtype

      data_shape = source_file.attrs['dims']

      fileout.create_dataset(name=dataset,
                             shape=data_shape,
                             dtype=dtype,
                             compression=compression_type,
                             compression_opts=compression_options)

  # loop over files for a given output
  for i in range(0, num_processes):
    # open the input file for reading
    filein = h5py.File(input_dir / f'{output_number}.h5.{i}', 'r')
    # read in the header data from the input file
    head = filein.attrs

    # write data from individual processor file to correct location in concatenated file
    nx_local, ny_local, nz_local = filein.attrs['dims_local']
    x_start, y_start, z_start    = filein.attrs['offset']

    for dataset in datasets_to_copy:
      fileout[dataset][x_start:x_start+nx_local, y_start:y_start+ny_local,z_start:z_start+nz_local] = filein[dataset]

    filein.close()

  fileout.close()
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

if __name__ == '__main__':
  main()
