#!/usr/bin/env python3
"""
Python script for concatenating particle hdf5 datasets. Includes a CLI for
concatenating Cholla HDF5 datasets and can be imported into other scripts
where the ``concat_particles_dataset`` function can be used to concatenate
the datasets.

Historically, this docstring provided advice for directly importing functionality from
this script.

* at this time, it's unclear whether anybody actually does this (it has become less
  necessary now that we provide tools to load in distributed data). If you actually use
  some part of this functionality, please open a GitHub issue letting us know so we can
  add that functionality directly into the ``cholla_utils`` python package (of course,
  we welcome you to open a PR making that change yourself)

* the approach we have historically recommended involves adding the ``python/scripts``
  directory directly the search path for python modules. To do this, you might add
  something like the following snippet to your python script:

    import sys
    sys.path.append('</PATH/TO/CHOLLA>/python/scripts')
    import concat_2particles

  where you would replace ``</PATH/TO/CHOLLA>`` with an absolute path to your Cholla
  directory.
"""

import h5py
from typing import Optional
import pathlib

# normally, it's considered bad practice to import a submodule starting with an
# underscore (since that submodule is considered an implementation detail), but the
# following is done for backwards compatability as we reorganize
import cholla_utils._concat_internals as concat_internals

from snaprepack import DatasetOpts, dset_opts_from_args, SnapBuilder


def concat_particles_dataset(
    output_directory: pathlib.Path,
    output_number: int,
    build_source_path,
    *,
    skip_fields: list = [],
    dset_opts: Optional[DatasetOpts] = None,
    ptype_name: Optional[str] = None,
) -> None:
    """Concatenate a single particle HDF5 Cholla dataset. i.e. take the single
    files generated per process and concatenate them into a single, large file.

    Parameters
    ----------
    output_directory : pathlib.Path
        The directory containing the new concatenated files
    output_number : int
        The output number to concatenate
    build_source_path : callable
        A function used to construct the paths to the files that are to be concatenated.
    skip_fields : list
        List of fields to skip concatenating. Defaults to [].
    dset_opts
        Optional kwargs for ``h5py.Group.create_dataset``.
    ptype_name: str, optional
        A name to use for the particle-type when no particle-type is recorded
    """

    src_path_0 = build_source_path(proc_id=0, nfile=output_number)

    # Error checking
    assert output_number >= 0, "output_number must be greater than or equal to 0"

    # we open up the file associated with source-file 0 to examine header details
    with h5py.File(src_path_0, "r") as src_f_0:
        num_files = concat_internals.infer_numfiles_from_header(src_f_0.attrs)
        if num_files < 2:
            raise RuntimeError(
                "it only makes sense to concatenate data split across 2 or more files"
            )

        # create a sequence over (blockid, fname) pairs. Note: the very 1st entry will
        # correspond to src_path_0, but that's totally okay
        itr = [
            (blockid, build_source_path(proc_id=blockid, nfile=output_number))
            for blockid in range(0, num_files)
        ]

        # right now this is hard-coded
        single_ptype_cholla_outputs = True

        # get the stored particle types and the number of particles of each type
        # -> in the future, this will hopefully be recorded in the hdf5 file
        if single_ptype_cholla_outputs:
            if ptype_name is None:
                raise ValueError(
                    "the ptype_name kwarg must be provided when no particle type is "
                    "recorded in the file"
                )
            # count up the total particle count
            num_particles = src_f_0.attrs["n_particles_local"]
            for _, path in itr[1:]:
                with h5py.File(path, "r") as f:
                    num_particles += f.attrs["n_particles_local"]
            total_ptype_counts = {ptype_name: num_particles}
        else:
            assert ptype_name is None
            raise NotImplementedError()

        # let's write the concatenated file
        with SnapBuilder(output_directory / f"{output_number}_particles.h5") as builder:
            builder.set_hdr(src_f_0).particle_config(
                total_ptype_counts=total_ptype_counts,
                concatenating_single_ptype_cholla_outputs=single_ptype_cholla_outputs,
                opts=dset_opts,
                skip=skip_fields,
            ).particle_record_itr(itr, file_paths=True).write()


# ==============================================================================

if __name__ == "__main__":
    from timeit import default_timer

    start = default_timer()

    cli = concat_internals.common_cli(num_processes_choice="omit")
    cli.add_argument("--ptype", default="io", help="the particle type to concatenate")
    args = cli.parse_args()

    dset_opts = dset_opts_from_args(args)
    build_source_path = concat_internals.get_source_path_builder(
        source_directory=args.source_directory,
        pre_extension_suffix="_particles",
        known_output_snap=args.concat_outputs[0],
    )

    # Perform the concatenation
    for output in args.concat_outputs:
        concat_particles_dataset(
            output_directory=args.output_directory,
            output_number=output,
            build_source_path=build_source_path,
            skip_fields=args.skip_fields,
            dset_opts=dset_opts,
            ptype_name=args.ptype,
        )

    print(f"\nTime to execute: {round(default_timer() - start, 2)} seconds")
