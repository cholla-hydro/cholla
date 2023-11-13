#!/usr/bin/env python3
"""
Python script for concatenating particle hdf5 datasets. Includes a CLI for concatenating Cholla HDF5 datasets and can be
imported into other scripts where the `concat_particles_dataset` function can be used to concatenate the datasets.

Generally the easiest way to import this script is to add the `python_scripts` directory to your python path in your
script like this:
```
import sys
sys.path.append('/PATH/TO/CHOLLA/python_scripts')
import concat_particles
```
"""

import h5py
import numpy as np
import pathlib

import concat_internals

# ======================================================================================================================
def concat_particles_dataset(source_directory: pathlib.Path,
                             output_directory: pathlib.Path,
                             num_processes: int,
                             output_number: int,
                             skip_fields: list = [],
                             destination_dtype: np.dtype = None,
                             compression_type: str = None,
                             compression_options: str = None,
                             chunking = None) -> None:
  """Concatenate a single particle HDF5 Cholla dataset. i.e. take the single
  files generated per process and concatenate them into a single, large file.

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
  destination_file = concat_internals.destination_safe_open(output_directory / f'{output_number}_particles.h5')

  # Setup the output file
  # Note that the call to `__get_num_particles` is potentially expensive as it
  # opens every single file to read the number of particles in that file
  num_particles    = __get_num_particles(source_directory, num_processes, output_number)
  destination_file = __setup_destination_file(source_directory,
                                              destination_file,
                                              output_number,
                                              num_particles,
                                              skip_fields,
                                              destination_dtype,
                                              compression_type,
                                              compression_options,
                                              chunking)

  # loop over files for a given output
  particles_offset = 0
  for i in range(0, num_processes):
    # open the input file for reading
    source_file = h5py.File(source_directory / f'{output_number}_particles.h5.{i}', 'r')

    # Compute the offset slicing for the 3D data
    nx_local, ny_local, nz_local = source_file.attrs['dims_local']
    x_start, y_start, z_start    = source_file.attrs['offset']
    x_end, y_end, z_end          = x_start+nx_local, y_start+ny_local, z_start+nz_local

    # Get the local number of particles
    num_particles_local = source_file.attrs['n_particles_local'][0]

    # write data from individual processor file to correct location in concatenated file
    for dataset in list(destination_file.keys()):

      if dataset == 'density':
        destination_file[dataset][x_start:x_end,
                                  y_start:y_end,
                                  z_start:z_end] = source_file[dataset]
      else:
        start = particles_offset
        end   = particles_offset + num_particles_local
        destination_file[dataset][start:end] = source_file[dataset]

    # Update the particles offset
    particles_offset += num_particles_local

    # Now that the copy is done we close the source file
    source_file.close()

  # Close destination file now that it is fully constructed
  destination_file.close()
# ==============================================================================

# ==============================================================================
def __get_num_particles(source_directory: pathlib.Path,
                        num_processes: int,
                        output_number: int) -> int:
  """Get the total number of particles in the output. This function is heavily
  I/O bound and might benefit from utilizing threads.

  Parameters
  ----------
  source_directory : pathlib.Path
      The directory of the unconcatenated files
  num_processes : int
      The number of processes
  output_number : int
      The output number to get data from

  Returns
  -------
  int
      The total number of particles in the output
  """
  # loop over files for a given output
  num_particles = 0
  for i in range(0, num_processes):
    # open the input file for reading
    with h5py.File(source_directory / f'{output_number}_particles.h5.{i}', 'r') as source_file:
      num_particles += source_file.attrs['n_particles_local']

  return num_particles
# ==============================================================================

# ==============================================================================
def __setup_destination_file(source_directory: pathlib.Path,
                             destination_file: h5py.File,
                             output_number: int,
                             num_particles: int,
                             skip_fields: list,
                             destination_dtype: np.dtype,
                             compression_type: str,
                             compression_options: str,
                             chunking) -> h5py.File:
  """_summary_

  Parameters
  ----------
  source_directory : pathlib.Path
      The directory containing the unconcatenated files
  destination_file : h5py.File
      The destination file
  output_number : int
      The output number to concatenate
  num_particles : int
      The total number of particles in the output
  skip_fields : list
      List of fields to skip concatenating.
  destination_dtype : np.dtype
      The data type of the output datasets. Accepts most numpy types.
  compression_type : str
      What kind of compression to use on the output data.
  compression_options : str
      What compression settings to use if compressing.
  chunking : _type_
      Whether or not to use chunking and the chunk size.

  Returns
  -------
  h5py.File
      The fully set up destination file
  """
  with h5py.File(source_directory / f'{output_number}_particles.h5.0', 'r') as source_file:
    # Copy header data
    destination_file = concat_internals.copy_header(source_file, destination_file)

    # Make list of datasets to copy
    datasets_to_copy = list(source_file.keys())
    datasets_to_copy = [dataset for dataset in datasets_to_copy if not dataset in skip_fields]

    # Create the datasets in the output file
    for dataset in datasets_to_copy:
      dtype = source_file[dataset].dtype if (destination_dtype == None) else destination_dtype

      # Determine the shape of the dataset
      if dataset == 'density':
        data_shape = source_file.attrs['dims']
      else:
        data_shape = num_particles

      # Create the dataset
      destination_file.create_dataset(name=dataset,
                                      shape=data_shape,
                                      dtype=dtype,
                                      chunks=chunking,
                                      compression=compression_type,
                                      compression_opts=compression_options)

  return destination_file
# ==============================================================================

if __name__ == '__main__':
  from timeit import default_timer
  start = default_timer()

  cli = concat_internals.common_cli()
  args = cli.parse_args()

  # Perform the concatenation
  for output in args.concat_outputs:
    concat_particles_dataset(source_directory=args.source_directory,
                             output_directory=args.output_directory,
                             num_processes=args.num_processes,
                             output_number=output,
                             skip_fields=args.skip_fields,
                             destination_dtype=args.dtype,
                             compression_type=args.compression_type,
                             compression_options=args.compression_opts,
                             chunking=args.chunking)

  print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
