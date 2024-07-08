#!/usr/bin/env python3
"""
Contains all the common tools for the various concatnation functions/scipts
"""

import h5py
import numpy as np

import argparse
import functools
import pathlib
import re
import warnings

# imports for type annotations:
from collections.abc import Mapping
from typing import Optional

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

def infer_numfiles_from_header(hdr: Mapping) -> int:
  """Infers the total number of ranks that cholla was run with to produce this
  file. Equivalently, this returns the number of files that must be
  concatenated.

  Parameters
  ----------
  hdr: Mapping
    ``dict``-like object specifying the core attributes of an hdf5 file. This
    is commonly the value returned by the ``attrs`` property of a ``h5py.File``
    instance (but we don't really care about the type).

  Returns
  -------
  int
    The number of files that must be concatenated

  Notes
  -----
  In the future, it would be nice to directly encode this information rather
  than requiring us to encode it
  """
  dims, dims_local = hdr['dims'], hdr['dims_local']
  assert np.issubdtype(dims.dtype, np.signedinteger) # sanity check
  assert np.issubdtype(dims_local.dtype, np.signedinteger) # sanity check

  blocks_per_ax, remainders = np.divmod(dims, dims_local, dtype = 'i8')
  assert np.all(blocks_per_ax > 0) and np.all(remainders == 0)  # sanity check

  return int(np.prod(blocks_per_ax))


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
  fields_to_skip = ['dims_local', 'offset', 'n_particles_local']

  for attr_key in source_file.attrs.keys():
    if attr_key not in fields_to_skip:
      destination_file.attrs[attr_key] = source_file.attrs[attr_key]

  return destination_file
# ==============================================================================

def _integer_sequence(s: str):
  # converts an argument string to an integer sequence
  # -> s can be a range specified as start:stop:step. This follows mirrors
  #    the semantics of a python slice (at the moment, start and stop are
  #    both required)
  # -> s can b a comma separated list
  # -> s can be a single value
  m = re.match(
    r"(?P<start>[-+]?\d+):(?P<stop>[-+]?\d+)(:(?P<step>[-+]?\d+))?",
    s)
  if m is not None:
    rslts = m.groupdict()
    step = 1
    if rslts['step'] is not None:
      step = int(rslts.get('step',1))
    if step == 0:
      raise ValueError(f"The range, {s!r}, has a stepsize of 0")
    seq = range(int(rslts['start']), int(rslts['stop']), step)
    if len(seq) == 0:
      raise ValueError(f"The range, {s!r}, has 0 values")
    return seq
  elif re.match(r"([-+]?\d+)(,[ ]*[-+]?\d+)+", s):
    seq = [int(elem) for elem in s.split(',')]
    return seq
  try:
    return [int(s)]
  except ValueError:
    raise ValueError(
      f"{s!r} is invalid. It should be a single int or a range"
    ) from None


# ==============================================================================
def _add_snaps_arg(cli, required: bool = False):
  cli.add_argument(
    '--snaps', type=_integer_sequence, dest = "concat_outputs",
    required = required,
    metavar='(NUM | START:STOP[:STEP] | N1,N2,...)',
    help = ('Specify output(s) to concatenate. Either a single number '
            '(e.g. 8), a range (in python slice syntax), or a list (e.g. '
            '1,2,3)')
  )

def add_common_cli_args(cli: argparse.ArgumentParser,
                        num_processes_choice: str,
                        add_concat_outputs_arg: bool = True):
  """Add common command-line arguments to an argparse.ArguementParser instance
  
  These arguments are shared among the various concatenation scripts.

  Parameters
  ----------
  cli: argparse.ArgumentParser
      Instance that arguments are added to
  """

  # ============================================================================
  def concat_output(raw_argument: str,)-> list:
    """Function used to parse the `--concat-output` argument
    """
    warnings.warn(
      "The -c/--concat-output flag is now deprecated. use the --snaps "
      "flag instead"
    )
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

    return list(iterable_argument)
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

  if num_processes_choice == 'use':
    cli.add_argument(
      '-n', '--num-processes',    type=positive_int,  required=True,
      help='The number of processes that were used while running Cholla.')
  elif num_processes_choice == 'deprecate':
    cli.add_argument(
      '-n', '--num-processes', type=positive_int, required=False,
      default = None,
      help='DEPRECATED: The number of processes that were used while running Cholla.')
  elif num_processes_choice != 'omit':
    raise ValueError('invalid value passed for num_processes_choice')

  if add_concat_outputs_arg:
    grp = cli.add_mutually_exclusive_group(required=True)
    grp.add_argument(
      '-c', '--concat-outputs',   type=concat_output,
      help = 'DEPRECATED (use --snaps instead) Specify outputs to concatenate. Can be a single number (e.g. 8), an inclusive range (e.g. 2-9), or a list (e.g. [1,2,3]).')
    _add_snaps_arg(grp, required = False)
  else:
    _add_snaps_arg(cli, required = True)

  # Other Required Arguments
  cli.add_argument('-s', '--source-directory', type=pathlib.Path,  required=True, help='The path to the directory for the source HDF5 files.')
  cli.add_argument('-o', '--output-directory', type=pathlib.Path,  required=True, help='The path to the directory to write out the concatenated HDF5 files.') 
    

  # Optional Arguments
  cli.add_argument('--skip-fields',            type=skip_fields,   default=[],   help='List of fields to skip concatenating. Defaults to empty.')
  cli.add_argument('--dtype',                  type=str,           default=None, help='The data type of the output datasets. Accepts most numpy types. Defaults to the same as the input datasets.')
  cli.add_argument('--compression-type',       type=str,           default=None, help='What kind of compression to use on the output data. Defaults to None.')
  cli.add_argument('--compression-opts',       type=str,           default=None, help='What compression settings to use if compressing. Defaults to None.')
  cli.add_argument('--chunking',               type=chunk_arg,     default=None, nargs='?', const=True, help='Enable chunking of the output file. Default is `False`. If set without an argument then the chunk size will be automatically chosen or a tuple can be passed to indicate the chunk size desired.')


def common_cli(num_processes_choice = 'use') -> argparse.ArgumentParser:
  """This function provides the basis for the common CLI amongst the various
  concatenation scripts.
  
  It returns a newly constructed `argparse.ArgumentParser` object to which
  additional arguments can be passed before the final `.parse_args()` method is
  used.

  Parameters
  ----------

  Returns
  -------
  argparse.ArgumentParser
    The common components of the CLI for the concatenation scripts
  """
  # Initialize the CLI
  cli = argparse.ArgumentParser()
  add_common_cli_args(cli, num_processes_choice = num_processes_choice,
                      add_concat_outputs_arg = True)
  return cli

# ==============================================================================

def _get_source_path(proc_id : int, source_directory : pathlib.Path,
                     pre_extension_suffix : str, nfile : int, new_style : bool,
                     extension : str = '.h5'):
  dirname = str(source_directory)
  if new_style:
    out = f"{dirname}/{nfile}/{nfile}{pre_extension_suffix}{extension}.{proc_id}"
  else:
    # in principle, when source_directory isn't an empty string and it doesn't end
    # end in a '/', part of it should act like a filename prefix
    # -> with that said, the concatenation scripts have not supported this behavior
    #    since we've made use of pathlib.Path
    out = f"{dirname}/{nfile}{pre_extension_suffix}{extension}.{proc_id}"
  return pathlib.Path(out)

def get_source_path_builder(source_directory : pathlib.Path,
                            pre_extension_suffix : str,
                            known_output_snap : int):
  """
  Source files (that are to be concatenated) have one of 2 formats. This identifies
  the format in use and returns a function appropriate for building the pathnames

  This function auto-detect the format and returns a function to construct paths to these
  files
  """

  # try newer format first:
  common_kw = {'source_directory' : source_directory, 'extension' : '.h5',
               'pre_extension_suffix' : pre_extension_suffix}
  new_style_path = _get_source_path(proc_id = 0, nfile = known_output_snap,
                                    new_style = True, **common_kw)
  old_style_path = _get_source_path(proc_id = 0, nfile = known_output_snap,
                                    new_style = False, **common_kw)
  if new_style_path.is_file():
    return functools.partial(_get_source_path, new_style = True, **common_kw)
  elif old_style_path.is_file():
    return functools.partial(_get_source_path, new_style = False, **common_kw)
  raise RuntimeError(
    "Could not find any files to concatenate. We searched "
    f"{new_style_path!s} and {old_style_path!s}"
  )