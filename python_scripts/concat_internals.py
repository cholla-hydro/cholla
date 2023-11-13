#!/usr/bin/env python3
"""
Contains all the common tools for the various concatnation functions/scipts
"""

import h5py
import argparse
import pathlib

# ==============================================================================
def destination_safe_open(filename: pathlib.Path) -> h5py.File:
  """Opens a HDF5 file safely and provides useful error messages for some common failure modes

  Parameters
  ----------
  filename : pathlib.Path

  The full path and name of the file to open :

  filename: pathlib.Path :


  Returns
  -------
  h5py.File

  The opened HDF5 file object
  """

  try:
    destination_file = h5py.File(filename, 'w-')
  except FileExistsError:
    # It might be better for this to simply print the error message and return
    # rather than exiting. That way if a single call fails in a parallel
    # environment it doesn't take down the entire job
    raise FileExistsError(f'File "{filename}" already exists and will not be overwritten, skipping.')

  return destination_file
# ==============================================================================

# ==============================================================================
def copy_header(source_file: h5py.File, destination_file: h5py.File) -> h5py.File:
  """Copy the attributes of one HDF5 file to another, skipping all fields that are specific to an individual rank

  Parameters
  ----------
  source_file : h5py.File
      The source file
  destination_file : h5py.File
      The destination file
  source_file: h5py.File :

  destination_file: h5py.File :


  Returns
  -------
  h5py.File
      The destination file with the new header attributes
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

  Parameters
  ----------

  Returns
  -------
  argparse.ArgumentParser
    The common components of the CLI for the concatenation scripts
  """

  # ============================================================================
  def concat_output(raw_argument: str) -> list:
    """Function used to parse the `--concat-output` argument
    """
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
  def chunk_arg(raw_argument: str) -> tuple:
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
