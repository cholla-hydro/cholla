#!/usr/bin/env python3
"""
Command line program that provides a unified interface for concatenating 3D
hdf5 data dumps produced by Cholla.

At the moment, we only support fluid quantities. In the future, we could
support other types of fields.
"""

import numpy as np

import argparse
import datetime
from functools import partial
import os
import pathlib

import concat_internals
from concat_2d_data import concat_2d_dataset
from concat_3d_data import concat_3d_dataset

parser = argparse.ArgumentParser(
    description = ("Concatenates HDF5 ouputs produced by Cholla")
)
concat_internals.add_common_cli_args(parser, num_processes_choice = 'omit',
                                     add_concat_outputs_arg = False)

_2D_kinds = ("proj", "slice", "rot_proj")

parser.add_argument("--kind", nargs = "+", required = True,
    help = (
        "specifies the types of hdf5 files that are to be concatenated. This "
        f"can be `3D` or the name of a 2D dataset `{'`, `'.join(_2D_kinds)}`. "
        "For a 2D dataset like 'proj', you can append a suffix (e.g. `-xy`, "
        "`-yz`, `-xz`) or a series of suffixes (e.g. `-xy,yz`) to specify "
        "only a subset of the datasets should be concatenated"))

def _try_identify_2D_kind_kwargs(kind):
    # try to identify the 2d dataset-kind and any associated kwargs for
    # concat_2D_dataset

    prefix = None
    for k in _2D_kinds:
        if kind.startswith(k):
            prefix = k
            break
    else: # this get's executed if we don't break out of for-loop
        return None

    suffix = kind[len(prefix):]
    tmp = {'concat_xy' : False, 'concat_yz' : False, 'concat_xz' : False}
    if suffix in ['', '-xy,yz,xz', '-xy,xz,yz', '-yz,xy,xz', '-xz,xy,yz',
                  '-yz,xz,xy', '-xz,yz,xy']:
        for key in tmp:
            tmp[key] = True
    elif suffix == '-xy':
        tmp['concat_xy'] = True
    elif suffix == '-xz':
        tmp['concat_xz'] = True
    elif suffix == '-yz':
        tmp['concat_yz'] = True
    elif suffix in ['-xy,xz', '-xz,xy']:
        tmp['concat_xy'] = True
        tmp['concat_xz'] = True
    elif kind in ['-xy,yz', '-yz,xy']:
        tmp['concat_xy'] = True
        tmp['concat_yz'] = True
    elif kind in ['-xz,yz', '-yz,xz']:  
        tmp['concat_xz'] = True
        tmp['concat_yz'] = True
    else:
        raise ValueError(f"{kind} has an invalid suffix")
    return prefix, tmp

def _handle_kinds_processing(kind_l: list):
    encountered_3D = False
    encountered_2D_kind_set = set()
    kindkw_2D_pairs = []

    for kind in kind_l:
        if kind == '3D':
            if encountered_3D:
                raise ValueError("3D kind appears more than once")
            encountered_3D = True
            continue
        # try to treat kind as 2D kind
        pair = _try_identify_2D_kind_kwargs(kind)
        if pair is not None:
            if pair[0] in encountered_2D_kind_set:
                raise ValueError(f"{kind} appears more than once")
            encountered_2D_kind_set.add(pair[0])
            kindkw_2D_pairs.append(pair)
        else:
            raise ValueError(f"{kind} is a totally unrecognized dataset-kind")
    return encountered_3D, kindkw_2D_pairs

def main(args):
    handle_3D, kindkw_2D_pairs = _handle_kinds_processing(args.kind)
    assert handle_3D or len(kindkw_2D_pairs) > 0 # sanity-check

    # create a function to build_source_paths
    if handle_3D:
        _temp_pre_extension_suffix = ''
    else:
        _temp_pre_extension_suffix = f'_{kindkw_2D_pairs[0][0]}'

    temp_build_source_path = concat_internals.get_source_path_builder(
        source_directory = args.source_directory,
        pre_extension_suffix = _temp_pre_extension_suffix,
        known_output_snap = args.concat_outputs[0])

    # construct a list of concatenation commands performed at each output
    command_triples = []
    if handle_3D:
        command_triples.append(
            ('3D',
             partial(temp_build_source_path, pre_extension_suffix = ''),
             concat_3d_dataset)
        )
    for kind_2D, kwargs in kindkw_2D_pairs:
        command_triples.append(
            (kind_2D,
             partial(temp_build_source_path,
                     pre_extension_suffix = f'_{kind_2D}'),
             partial(concat_2d_dataset, dataset_kind=kind_2D, **kwargs))
        )

    #raise RuntimeError(repr(command_triples))

    # create the output directory if it doesn't already exist...
    abs_out_dir = os.path.abspath(args.output_directory)
    if not os.path.exists(abs_out_dir):
        if os.path.exists(os.path.dirname(abs_out_dir)):
            os.mkdir(abs_out_dir)
        else:
            raise RuntimeError(
                f"Can't create {args.output_directory} since the "
                f"{args.output_directory}/.. directory doesn't already exist")

    for output in args.concat_outputs:
        print(f"concatenating {output}")
        for dset_kind, build_source_path, concat_fn in command_triples:
            t1 = datetime.datetime.now()
            concat_fn(output_directory=args.output_directory,
                      output_number=output,
                      build_source_path = build_source_path,
                      skip_fields=args.skip_fields,
                      destination_dtype=args.dtype,
                      compression_type=args.compression_type,
                      compression_options=args.compression_opts,
                      chunking=args.chunking)
            t2 = datetime.datetime.now()
            print(f'  -> {dset_kind!r}: {(t2 - t1).total_seconds()}')



if __name__ == '__main__':
    main(parser.parse_args())
