#!/usr/bin/env python3
"""
Python script for concatenating 3D hdf5 datasets. Includes a CLI for concatenating Cholla HDF5 datasets and can be
imported into other scripts where the `concat_3d_dataset` function can be used to concatenate the datasets.

Generally the easiest way to import this script is to add the `python_scripts` directory to your python path in your
script like this:
```
import sys
sys.path.append('/PATH/TO/CHOLLA/python_scripts')
import concat_3d_data
```
"""

import h5py

import os
import pathlib
from typing import Optional
import warnings

import concat_internals
from snaprepack import DatasetOpts, dset_opts_from_args, SnapBuilder


# ==============================================================================
def _check_num_processes_arg(path: os.PathLike, num_processes_arg: int):
    with h5py.File(path, "r") as f:
        num_files = concat_internals.infer_numfiles_from_header(f.attrs)
        if num_processes_arg != num_files:
            raise RuntimeError(
                f"header of {path!r} implies that it contains a subset of data that was split "
                f"across {num_files} files (rather than across {num_processes_arg} files)."
            )
    return num_files


def concat_3d_dataset(
    output_directory: pathlib.Path,
    output_number: int,
    build_source_path,
    *,
    num_processes: Optional[int] = None,
    skip_fields: list = [],
    dset_opts: Optional[DatasetOpts] = None,
    legacy_field: bool = False,
) -> None:
    """Concatenate a single 3D HDF5 Cholla dataset. i.e. take the single files
    generated per process and concatenate them into a single, large file.

    Parameters
    ----------
    output_directory : pathlib.Path
        The directory containing the new concatenated files
    num_processes : int
        The number of ranks that Cholla was run with
    output_number : int
        The output number to concatenate
    skip_fields : list
        List of fields to skip concatenating. Defaults to [].
    build_source_path : callable
        A function used to construct the paths to the files that are to be concatenated.
    num_processes : int, optional
        The number of ranks that Cholla was run with. This information is now inferred
        from the hdf5 file and the parameter will be removed in the future.
    dset_opts
        Optional kwargs for ``h5py.Group.create_dataset``.
    legacy_field
        When True, the "field_legacy" HDF5 group will be created.
    """

    src_path_0 = build_source_path(proc_id=0, nfile=output_number)

    # Error checking
    assert output_number >= 0, "output_number must be greater than or equal to 0"
    if num_processes is not None:
        warnings.warn("the num_processes parameter will be removed")
        _check_num_processes_arg(src_path_0, num_processes_arg=num_processes)

    # we open up the file associated with source-file 0 to examine header details
    with h5py.File(src_path_0, "r") as src_f_0:
        num_files = concat_internals.infer_numfiles_from_header(src_f_0.attrs)
        if num_files < 2:
            raise RuntimeError(
                "it only makes sense to concatenate data split across 2 or more files"
            )

        # create iterator over (blockid, fname) pairs. Note: the very 1st entry will
        # correspond to src_path_0, but that's totally okay
        itr = (
            (blockid, build_source_path(proc_id=blockid, nfile=output_number))
            for blockid in range(0, num_files)
        )

        # let's write the concatenated file
        with SnapBuilder(output_directory / f"{output_number}.h5") as builder:
            builder.set_hdr(src_f_0).field_config(
                opts=dset_opts, skip=skip_fields, legacy=legacy_field
            ).field_record_itr(itr, file_paths=True).write()


# ==============================================================================

if __name__ == "__main__":
    from timeit import default_timer

    start = default_timer()

    cli = concat_internals.common_cli(num_processes_choice="deprecate")
    concat_internals._add_legacyfield_arg(cli)
    args = cli.parse_args()

    dset_opts = dset_opts_from_args(args)
    build_source_path = concat_internals.get_source_path_builder(
        source_directory=args.source_directory,
        pre_extension_suffix="",
        known_output_snap=args.concat_outputs[0],
    )

    # Perform the concatenation
    for output in args.concat_outputs:
        concat_3d_dataset(
            output_directory=args.output_directory,
            num_processes=args.num_processes,
            output_number=output,
            build_source_path=build_source_path,
            skip_fields=args.skip_fields,
            dset_opts=dset_opts,
            legacy_field=args.legacy_field,
        )

    print(f"\nTime to execute: {round(default_timer() - start, 2)} seconds")
