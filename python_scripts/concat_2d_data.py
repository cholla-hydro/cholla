#!/usr/bin/env python3
"""
Python script for concatenating 2D hdf5 datasets for when -DSLICES,
-DPROJECTION, or -DROTATED_PROJECTION is turned on in Cholla. Includes a CLI for
concatenating Cholla HDF5 datasets and can be imported into other scripts where
the `concat_2d_dataset` function can be used to concatenate the HDF5 files.

Generally the easiest way to import this script is to add the `python_scripts`
directory to your python path in your script like this:
```
import sys
sys.path.append('/PATH/TO/CHOLLA/python_scripts')
import concat_2d_data
```
"""

import h5py
import pathlib
import numpy as np

import concat_internals

# ==============================================================================
def concat_2d_dataset(output_directory: pathlib.Path,
                      num_processes: int,
                      output_number: int,
                      dataset_kind: str,
                      build_source_path,
                      concat_xy: bool = True,
                      concat_yz: bool = True,
                      concat_xz: bool = True,
                      skip_fields: list = [],
                      destination_dtype: np.dtype = None,
                      compression_type: str = None,
                      compression_options: str = None,
                      chunking = None) -> None:
  """Concatenate 2D HDF5 Cholla datasets. i.e. take the single files
    generated per process and concatenate them into a single, large file. This
    function concatenates a single output time and can be called multiple times,
    potentially in parallel, to concatenate multiple output times.

  Parameters
  ----------
  output_directory : pathlib.Path
      The directory containing the new concatenated files
  num_processes : int
      The number of ranks that Cholla was run with
  output_number : int
      The output number to concatenate
  dataset_kind : str
      The type of 2D dataset to concatenate. Can be 'slice', 'proj', or 'rot_proj'.
  build_source_path : callable
      A function used to construct the paths to the files that are to be concatenated.
  concat_xy : bool
      If True then concatenate the XY slices/projections. Defaults to True.
  concat_yz : bool
      If True then concatenate the YZ slices/projections. Defaults to True.
  concat_xz : bool
      If True then concatenate the XZ slices/projections. Defaults to True.
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
  output_directory: pathlib.Path :

  num_processes: int :

  output_number: int :

  dataset_kind: str :

  concat_xy: bool :
        (Default value = True)
  concat_yz: bool :
        (Default value = True)
  concat_xz: bool :
        (Default value = True)
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
  assert dataset_kind in ['slice', 'proj', 'rot_proj'], '`dataset_kind` can only be one of "slice", "proj", "rot_proj".'

  # Open destination file
  destination_file = concat_internals.destination_safe_open(output_directory / f'{output_number}_{dataset_kind}.h5')

  # Setup the destination file
  with h5py.File(build_source_path(proc_id = 0, nfile = output_number), 'r') as source_file:
    # Copy over header
    destination_file = concat_internals.copy_header(source_file, destination_file)

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
    zero_array = np.zeros(1)
    for dataset in datasets_to_copy:
      dtype = source_file[dataset].dtype if (destination_dtype == None) else destination_dtype

      dataset_shape = __get_2d_dataset_shape(source_file, dataset)

      # Create array to initialize data to zero, this is required for projections
      if zero_array.shape != dataset_shape:
        zero_array = np.zeros(dataset_shape)

      destination_file.create_dataset(name=dataset,
                                      shape=dataset_shape,
                                      data=zero_array,
                                      dtype=dtype,
                                      chunks=chunking,
                                      compression=compression_type,
                                      compression_opts=compression_options)

  # Copy data
  for rank in range(num_processes):
    # Open source file
    source_file = h5py.File(build_source_path(proc_id = rank, nfile = output_number), 'r')

    # Loop through and copy datasets
    for dataset in datasets_to_copy:
      # Determine locations and shifts for writing
      (i0_start, i0_end, i1_start, i1_end), file_in_slice = __write_bounds_2d_dataset(source_file, dataset)

      # If this is a slice dataset we can skip loading the source file if that
      # file isn't in the slice
      if dataset_kind == 'slice' and not file_in_slice:
        continue

      # Copy the data, the summation is required for projections but not slices
      destination_file[dataset][i0_start:i0_end,
                                i1_start:i1_end] += source_file[dataset]

    # Now that the copy is done we close the source file
    source_file.close()

  # Close destination file now that it is fully constructed
  destination_file.close()
# ==============================================================================

# ==============================================================================
def __get_2d_dataset_shape(source_file: h5py.File, dataset: str) -> tuple:
  """Determine the shape of the full 2D dataset

  Args:
      source_file (h5py.File): The source file the get the shape information from
      dataset (str): The dataset to get the shape of

  Raises:
      ValueError: If the dataset name isn't a 2D dataset name

  Returns:
      tuple: The dimensions of the dataset
  """

  if 'xzr' in dataset:
    return (source_file.attrs['nxr'][0], source_file.attrs['nzr'][0])

  nx, ny, nz = source_file.attrs['dims']
  if 'xy' in dataset:
    dimensions = (nx, ny)
  elif 'yz' in dataset:
    dimensions = (ny, nz)
  elif 'xz' in dataset:
    dimensions = (nx, nz)
  else:
    raise ValueError(f'Dataset "{dataset}" is not a slice.')

  return dimensions
# ==============================================================================

# ==============================================================================
def __write_bounds_2d_dataset(source_file: h5py.File, dataset: str) -> tuple:
  """Determine the bounds of the concatenated file to write to

  Args:
      source_file (h5py.File): The source file to read from
      dataset (str): The name of the dataset to read from the source file

  Raises:
      ValueError: If the dataset name isn't a 2D dataset name

  Returns:
      tuple: The write bounds for the concatenated file to be used like
      `output_file[dataset][return[0]:return[1], return[2]:return[3]]` followed by a bool to indicate if the file is
      in the slice if concatenating a slice
  """

  if 'xzr' in dataset:
    return (source_file.attrs['nx_min'][0], source_file.attrs['nx_max'][0],
            source_file.attrs['nz_min'][0], source_file.attrs['nz_max'][0]), True

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
    raise ValueError(f'Dataset "{dataset}" is not a slice or projection.')

  return bounds, file_in_slice
# ==============================================================================

if __name__ == '__main__':
  from timeit import default_timer
  start = default_timer()

  cli = concat_internals.common_cli()
  cli.add_argument('-d', '--dataset-kind', type=str, required=True,    help='What kind of 2D dataset to concatnate. Options are "slice", "proj", and "rot_proj"')
  cli.add_argument('--disable-xy', default=True, action='store_false', help='Disables concating the XY datasets.')
  cli.add_argument('--disable-yz', default=True, action='store_false', help='Disables concating the YZ datasets.')
  cli.add_argument('--disable-xz', default=True, action='store_false', help='Disables concating the XZ datasets.')
  args = cli.parse_args()

  build_source_path = concat_internals.get_source_path_builder(
    source_directory = args.source_directory,
    pre_extension_suffix = f'_{args.dataset_kind}',
    known_output_snap = args.concat_outputs[0])

  # Perform the concatenation
  for output in args.concat_outputs:
    concat_2d_dataset(output_directory=args.output_directory,
                      num_processes=args.num_processes,
                      output_number=output,
                      dataset_kind=args.dataset_kind,
                      build_source_path = build_source_path,
                      concat_xy=args.disable_xy,
                      concat_yz=args.disable_yz,
                      concat_xz=args.disable_xz,
                      skip_fields=args.skip_fields,
                      destination_dtype=args.dtype,
                      compression_type=args.compression_type,
                      compression_options=args.compression_opts,
                      chunking=args.chunking)

  print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
