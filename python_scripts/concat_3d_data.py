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
import pathlib

import concat_internals

# ======================================================================================================================
def concat_3d_output(source_directory: pathlib.Path,
                     output_directory: pathlib.Path,
                     num_processes: int,
                     output_number: int,
                     skip_fields: list = [],
                     destination_dtype: np.dtype = None,
                     compression_type: str = None,
                     compression_options: str = None,
                     chunking = None) -> None:
  """Concatenate a single 3D HDF5 Cholla dataset. i.e. take the single files generated per process and concatenate them into a
    single, large file.

  Parameters
  ----------
  source_directory : pathlib.Path
      The directory containing the unconcatenated files
  output_directory : pathlib.Path
      The directory containing the new concatenated files
  num_processes : int
      The number of ranks that Cholla was run with
  output_number : int
      The output number to concatenate
  skip_fields : list
      List of fields to skip concatenating. Defaults to [].
  destination_dtype : np.dtype
      The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.
  compression_type : str
      What kind of compression to use on the output data. Defaults to None.
  compression_options : str
      What compression settings to use if compressing. Defaults to None.
  chunking : bool or tuple
      Whether or not to use chunking and the chunk size. Defaults to None.
  source_directory: pathlib.Path :

  output_directory: pathlib.Path :

  num_processes: int :

  output_number: int :

  skip_fields: list :
        (Default value = [])
  destination_dtype: np.dtype :
        (Default value = None)
  compression_type: str :
        (Default value = None)
  compression_options: str :
        (Default value = None)

  Returns
  -------

  """

  # Error checking
  assert num_processes > 1, 'num_processes must be greater than 1'
  assert output_number >= 0, 'output_number must be greater than or equal to 0'

  # Open the output file for writing
  destination_file = concat_internals.destination_safe_open(output_directory / f'{output_number}.h5')

  # Setup the output file
  with h5py.File(source_directory / f'{output_number}.h5.0', 'r') as source_file:
    # Copy header data
    destination_file = concat_internals.copy_header(source_file, destination_file)

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
# ==============================================================================

if __name__ == '__main__':
  from timeit import default_timer
  start = default_timer()

  cli = concat_internals.common_cli()
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
