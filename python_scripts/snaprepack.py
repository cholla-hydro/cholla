#!/usr/bin/env python3
"""
This file provides machinery to help build Cholla snapshots in postprocess-v2
format. A snapshot is any dataset that is a snapshot of the simulation state
(and is required for restarts). For example: fields, particles, gravity. This
does NOT include slices/projections
"""
import argparse
from collections import UserDict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import itertools
import os
import shutil
import sys
from typing import Any, Callable, Iterable, Optional, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
import h5py

import concat_internals

if sys.version_info >= (3, 11, 0):
    from typing import Self
else:
    Self = Any

int3 = tuple[int,int,int]


class DatasetOpts(TypedDict, total=False):
    # tracks kwargs for h5py.Group.create_dataset

    # The data type of the output datasets. Accepts most numpy types.
    dtype: np.dtype
    # kind of compression to use on the output data (if any)
    compression: str
    # Denotes compression settings (if any)
    compression_opts: str
    # Whether to use chunking and the chunk size
    chunks: Any


def _get_global_dims(f: h5py.File, f_is_concatenated:Optional[bool] = None):
    if "dims" in f.attrs:
        tmp = f.attrs["dims"]
        assert all(int(e) == e for e in tmp)
        return tuple(int(e) for e in tmp)
    elif f_is_concatenated and "density" in f:
        return f["density"].shape
    raise RuntimeError("can't infer global dims")


def _get_nprocs(f:h5py.File, nprocs: Optional[int3]=None) -> int3:
    f_has_nprocs = "nprocs" in f.attrs
    if nprocs is None and not f_has_nprocs:
        if "domain/blockid_location_arr" in f:
            return f["domain/blockid_location_arr"].shape
        raise ValueError("nprocs kwarg is required; it isn't an attr of f")
    elif nprocs is None:
        _nprocs = np.asarray(f.attrs["nprocs"])
    else:
        _nprocs = np.asarray(nprocs)
        if f_has_nprocs and np.any(_nprocs != np.asarray(f.attrs["nprocs"])):
            raise ValueError("nprocs kw inconsistent with f.attrs['nprocs']")

    # sanity checks
    assert np.all(_nprocs > 0) and len(_nprocs) == 3
    dims_local, remainders = np.divmod(f.attrs["dims"], _nprocs)
    if np.any(dims_local == 0) or np.any(remainders != 0):
        raise ValueError("global dims and nprocs are inconsistent")
    return tuple(_nprocs)


def _make_blockid_location_arr(
    f:h5py.File, *, nprocs: Optional[int3]=None
) -> NDArray[np.int64]:
    """Create a blockid_location_arr instance.

    As we note in the description of the file format, the blockid_location_arr
    is an array specifying the locations of each block. Negative value in the
    output denote missing values
    
    Parameters
    ----------
    f
        source file to try to read values from
    nprocs
        must be provided if the information isn't part source_file
    """

    nprocs = _get_nprocs(f=f, nprocs=nprocs)
    if "domain/blockid_location_arr" in f:
        assert f["domain/blockid_location_arr"].shape == nprocs
        return f["domain/blockid_location_arr"][...]
    else:
        shape = tuple(int(e) for e in nprocs)
        return np.full(shape=shape, fill_value=-1, dtype='i8')

def _calc_block_location(global_cell_offset:int3, block_cell_shape:int3) -> int3:
    """Compute the block's location in the blockid_location_arr

    Parameters
    ----------
    global_cell_offset
        In the conceptual global grid of cell-centered field-values, this
        denotes the 3D index of the leftmost cell in the considered block.
    block_cell_shape
        The shape of the array for storing a cell-centered field in each block
    Returns
    -------
    tuple
        An index of blockid_location_arr
    """
    out, remainders = np.divmod(global_cell_offset, block_cell_shape)
    assert np.all(out >= 0) and np.all(remainders==0)
    return tuple(out)


def _record_field_data(
    dst_f: h5py.File,
    expected_store_block_count: int,
    dest_idx: Any,
    data: Mapping,
    skip_fields: list,
    dset_opts: Mapping
) -> bool:
    # get the field names to be copied
    names = [name for name in data.keys() if name not in skip_fields]

    # get the "field" group (create it if it doesn't exist yet)
    if "field" not in dst_f:
        field_grp = dst_f.create_group("field")
        for field_name, dset in data.items():
            assert len(dset.shape) == 3 # sanity check!
            shape = (expected_store_block_count,) + dset.shape
            field_grp.create_dataset(name=field_name, shape=shape, **dset_opts)
        created_grp = True
    else:
        field_grp = dst_f["field"]
        created_grp = False

    # now, store the field data
    assert len(names) == len(field_grp) # sanity check!
    dest_sel = np.s_[dest_idx]
    for name in names:
        field_grp[name].write_direct(data[name], dest_sel=dest_sel)

    return created_grp

class SnapBuilder:
    """
    A snapshot-file builder, provideing fine-grained control over what is
    included in the file.

    This can be used as a context manager, (i.e. in a ``with``-statement), to
    help with proper cleanup if any errors occur

        
    Examples
    --------
    The public methods all return ``self`` to allow chaining.
    >>> with SnapBuilder(path, *args) as builder:
    ...     builder.set_hdr(src_f) \
    ...         .record_field_data_itr(itr) \
    ...         .write()
    """

    # note: the presence of type annotations means these are instance variables
    #       (not class variables)

    # attributes tracking global-file props
    _path: str         # path of file that we will create
    _tmp_path: str     # path to temporary file
    _f: h5py.File      # File object that represents the file @ _tmp_path
    _all_blocks: bool  # store all or 1 block in the file?

    _field_dset_opts: DatasetOpts  # kwargs for h5py.Group.create_dataset
    _skip_fields: set[str]         # fields that should be skipped

    # properties related to the domain group
    _stored_blockid_list: list[int]
    _blockid_location_arr: Optional[np.ndarray] = None
    _block_cell_shape: Optional[int3]  # used to compute block-locations

    def __init__(
        self,
        path: os.PathLike,
        all_blocks: bool,
        *,
        dset_opts: Optional[DatasetOpts]=None,
        skip_fields: Optional[list]=None
    ):
        self._path = os.fsdecode(path)
        self._tmp_path = f"{self._path}-tmp"
        assert not os.path.exists(self._path)
        assert not os.path.exists(self._tmp_path)
        self._f = h5py.File(self._tmp_path, "w")
        self._all_blocks = all_blocks

        self._field_dset_opts = dict() if dset_opts is None else dset_opts
        self._skip_fields = set() if skip_fields is None else set(skip_fields)

        self._stored_blockid_list = []
        self._blockid_location_arr = None
        self._block_cell_shape = None

    def _get_expected_local_block_count(self) -> int:
        if self._blockid_location_arr is None:
            raise RuntimeError("set_hdr method was never called")
        return self._blockid_location_arr.size if self._all_blocks else 1

    def set_hdr(
        self,
        source_file: Union[os.PathLike,h5py.File],
        *,
        nprocs: Optional[int3] = None
    ) -> Self:
        """Copy the header-attributes from source_file

        Kwargs
        ------
        nprocs
            This must be provided if the information isn't part source_file
        """
        if len(self._f.attrs) != 0:
            raise RuntimeError("It's an error to call set_hdr more than once")

        if isinstance(source_file, h5py.File):
            cm = nullcontext(source_file)
        else:
            cm = h5py.File(source_file, "r")

        with cm as src_f:
            # copy most of the header
            concat_internals.copy_header(src_f, self._f, skip_keys = ["nprocs"])

            # construct the array
            self._blockid_location_arr = _make_blockid_location_arr(
                f=src_f, nprocs=nprocs
            )

        nprocs = np.array(self._blockid_location_arr.shape)
        self._f.attrs["nprocs"] = nprocs

        global_dims = _get_global_dims(src_f)
        self._block_cell_shape, _remainder = np.divmod(global_dims, nprocs)
        assert np.all(self._block_cell_shape > 0) and np.all(_remainder==0)
        assert len(self._block_cell_shape) == 3 and len(nprocs) == 3

        return self

    def record_blockid_loc(self, blockid:int, global_cell_offset: int3) -> Self:
        """record the specified block's location

        Parameters
        ----------
        blockid
            The block's id
        global_cell_offset
            Location of the leftmost cell of the block in the conceptual grid
            spanning the entire domain
        """
        if self._block_cell_shape is None:
            raise RuntimeError("set_hdr method was never called")

        index = _calc_block_location(global_cell_offset, self._block_cell_shape)
        if self._blockid_location_arr[index] < 0:
            assert blockid not in self._stored_blockid_list  # sanity check
            assert blockid not in self._blockid_location_arr  # sanity check
            self._blockid_location_arr[index] = blockid
        else:
            assert self._blockid_location_arr[index] == blockid
        return Self

    def record_field_data(
        self,
        blockid: int,
        data: Mapping,
        *,
        global_cell_offset: Optional[int3] = None
    ) -> Self:
        """Record the specified block's field info

        Parameters
        ----------
        blockid
            The block's id
        data
            A dict-like object mapping field names to field data
        global_cell_offset
            Optionally specifies the location of the leftmost cell of the block
            in the conceptual grid spanning the entire domain. An error will be
            raised if this isn't specified & the block's location isn't known.
        """

        assert blockid >= 0
        if blockid in self._stored_blockid_list:
            raise RuntimeError("record_field_data already called for blockid")
        if global_cell_offset is not None:
            self.record_blockid_loc(blockid, global_cell_offset)

        # handles case where `data` is a h5py.File for existing snapshot file
        data = data["field"] if "field" in data else data

        newly_created = _record_field_data(
            dst_f=self._f,
            expected_store_block_count=self._get_expected_local_block_count(),
            dest_idx=(len(self._stored_blockid_list), ...),
            data=data,
            skip_fields=self._skip_fields,
            dset_opts=self._field_dset_opts
        )

        self._stored_blockid_list.append(blockid)

        if newly_created:
            assert self._block_cell_shape is not None  # sanity check!
            self._f["field"].attrs["block_cell_shape"] = self._block_cell_shape

        return self

    def record_field_data_itr(self, itr: Iterable) -> Self:
        """Calls self.record_field_data for each entry in itr"""

        for pack in itr:
            global_cell_offset = None
            if len(pack) == 2:
                blockid, data = pack
            elif len(pack) == 3:
                blockid, data, global_cell_offset = pack
            else:
                raise ValueError("each element of itr must be 2 or 3 elements")
            self.record_field_data(
                blockid=blockid, data=data, global_cell_offset=global_cell_offset
            )
        return self

    def write(self):
        # set require_all_blocks to False to write file with a subset of blocks
        if self._f is None:
            raise RuntimeError("already cleaned up the builder")

        # more sanity checks
        total_block_count = self._blockid_location_arr.size
        local_block_count = len(self._stored_blockid_list)
        if self._blockid_location_arr is None or len(self._f.attrs) == 0:
            raise RuntimeError("set_hdr was never called")
        elif np.any(self._blockid_location_arr < 0):
            raise RuntimeError("block location info is missing")
        elif local_block_count == 0:
            raise RuntimeError("no snapshot-data is stored for any block")
        elif self._all_blocks and total_block_count != local_block_count:
            raise RuntimeError("snapshot-data is missing for some blocks")

        # write the domain group of the file
        domain_grp = self._f.create_group("domain")
        domain_grp.create_dataset(
            "blockid_location_arr", data=self._blockid_location_arr
        )
        domain_grp.create_dataset(
            "stored_blockid_list", data=np.array(self._stored_blockid_list)
        )

        # save and close!
        self._f.close()
        self._f = None
        shutil.move(self._tmp_path, self._path)

    def cleanup(self):
        if self._f is not None:
            self._f.close()
            self._f = None
            os.remove(self._tmp_path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs):
        self.cleanup()

##########################################
class _MockBlockFieldGroup(UserDict):
    # acts like a h5py.Group instance for a selected region. This is better
    # than using a dict of loaded fields since that uses a lot more memory

    def __init__(self, grp: Mapping, idx_map: Mapping):
        super().__init__(dict(grp.items()))
        self._idx_map = idx_map

    def __getitem__(self, key:str):
        return self.data[key][self._idx_map.get(key,...)]

    def __repr__(self): return "<_MockBlockFieldGroup>"

def _iter_block_from_concat(
    src_f: h5py.File, missing_nprocs_triple: Optional[Sequence] = None,
)-> Iterable[tuple[int, _MockBlockFieldGroup, int3]]:
    """
    Iterates over blocks from a previously concatenated field

    Yields
    ------
    blockid: int
        The blockid
    data: _MockBlockFieldGroup
        maps field names to associated data
    global_cell_offset: tuple
        Location of the leftmost cell of the block in the conceptual grid
        spanning the entire domain
    """

    if "field" in src_f: # we dealing with file put into the new format
        assert missing_nprocs_triple is None

        blockid_list = src_f["domain/stored_blockid_list"][...]
        blockid_location_arr = src_f["domain/blockid_location_arr"][...]
        assert blockid_location_arr.size == blockid_list.size  # sanity check
        mapping = dict(zip(blockid_list, range(blockid_list.size)))

        grp = src_f["field"]
        stacked_field_shape = grp["density"].shape
        assert len(stacked_field_shape) == 4  # check invariant
        assert stacked_field_shape[0] == blockid_list.size  # check invariant
        block_cell_shape = stacked_field_shape[1:]

        for block_loc_idx, blockid in np.ndenumerate(blockid_location_arr):
            stack_idx = mapping[blockid]
            idx_map = {name: (stack_idx, ...) for name in grp}
            data = _MockBlockFieldGroup(grp, idx_map)

            global_cell_offset = tuple(
                block_loc_idx[i] * block_cell_shape[i] for i in range(3)
            )
            yield blockid, data, global_cell_offset

    else:
        nprocs = _get_nprocs(f=src_f, nprocs=missing_nprocs_triple)
        global_dims = _get_global_dims(src_f, True)
        block_cell_shape, _remainder = np.divmod(global_dims, nprocs)
        assert np.all(block_cell_shape > 0) and np.all(_remainder==0)
        assert len(block_cell_shape) == 3 and len(nprocs) == 3

        offset_itr = itertools.product(
            range(0, global_dims[0], block_cell_shape[0]),
            range(0, global_dims[1], block_cell_shape[1]),
            range(0, global_dims[2], block_cell_shape[2])    
        )

        def _mk_idx(name, start):
            shape = [e for e in block_cell_shape]
            if name == "magnetic_x":
                shape[0]+=1
            elif name == "magnetic_y":
                shape[1]+=1
            elif name == "magnetic_z":
                shape[2]+=1
            return tuple(slice(start[i], start[i]+shape[i], 1) for i in range(3))

        for blockid, global_cell_offset in enumerate(offset_itr):
            idx_map = {name: _mk_idx(name, global_cell_offset) for name in src_f}
            data = _MockBlockFieldGroup(src_f, idx_map)
            yield blockid, data, global_cell_offset

def repack_snapshot(
    out_dir: os.PathLike,
    src_path: os.PathLike,
    *,
    missing_nprocs_triple: Optional[Sequence] = None,
    skip_fields: Optional[Sequence] = None,
    dset_opts: Optional[DatasetOpts] = None,
):
    """This repacks an already concatenated snapshot

    Parameters
    ----------
    out_dir
        Path to directory where output files are written
    src_path
        Path to the input file
    missing_nprocs_triple
        Must be provided when the input file is missing the "nprocs" attr
    skip_fields
        Optional list of fields to skip concatenating.
    dset_opts
        Optional kwargs for ``h5py.Group.create_dataset``.
    """

    # coerce arguments
    if isinstance(skip_fields, str):
        raise TypeError("skip_fields can't be a string")
    elif missing_nprocs_triple is None or len(missing_nprocs_triple) == 0:
        missing_nprocs_triple = None
    else:
        assert len(missing_nprocs_triple) == 3
        assert all(elem > 0 for elem in missing_nprocs_triple)

    dset_opts = dict() if dset_opts is None else dset_opts

    out_path = os.path.join(out_dir, os.path.basename(src_path))
    if not os.path.isfile(src_path):
        raise ValueError(f"{src_path} doesn't exist")
    elif not os.path.isdir(out_dir):
        raise ValueError(f"{out_dir=} isn't a directory")
    elif os.path.exists(out_path):
        raise ValueError(f"{out_path} already exists")

    with h5py.File(src_path, "r") as src_f: # open the source file
        # do some basic setup
        itr = _iter_block_from_concat(src_f, missing_nprocs_triple)

        with SnapBuilder(
            out_path, True, dset_opts=dset_opts, skip_fields=skip_fields
        ) as builder: # open the snapshot-builder 
            builder.set_hdr(src_f, nprocs=missing_nprocs_triple) \
                .record_field_data_itr(itr) \
                .write()

parser = argparse.ArgumentParser(
    description = "Repacks (previously concatenated) HDF5 snapshots for Cholla"
)
concat_internals.add_common_cli_args(
    parser,
    num_processes_choice = 'omit',
    add_concat_outputs_arg = False,
    add_src_dir_arg=False
)
parser.add_argument(
    "--missing-nprocs-triple",
    nargs=3,
    default=None,
    type=int,
    help=(
        "This must be supplied when the 'nproc' attribute is missing from the "
        "input file. This specifies the number of processes along each axis."
    )
)

parser.add_argument(
    "--input-path-template",
    required=True,
    help=(
        "template-path for the input file. This path should generally include "
        "{snap}, which will be replaced by the snapshot numbers"
    )
)

if __name__ == '__main__':
    args = parser.parse_args()

    dset_opts = {
        "dtype" : args.dtype,
        "compression" : args.compression_type,
        "compression_opts" : args.compression_opts,
        "chunks" : args.chunking
    }

    for snap_number in args.concat_outputs:
        repack_snapshot(
            out_dir=args.output_directory,
            src_path=args.input_path_template.format(snap=snap_number),
            missing_nprocs_triple=args.missing_nprocs_triple,
            skip_fields=args.skip_fields,
            dset_opts=dset_opts
        )
