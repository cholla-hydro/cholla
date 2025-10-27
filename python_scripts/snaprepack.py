#!/usr/bin/env python3
"""
This file provides machinery to help build Cholla snapshots in the hierarichal
format. Specifically, this file can be invoked from the command line to repack
an existing file
"""

import argparse
from collections import UserDict
from contextlib import nullcontext
import errno
import functools
import itertools
import os
import shutil
import sys
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

import numpy as np
import h5py

import concat_internals

if sys.version_info >= (3, 11, 0):
    from typing import Self
else:
    Self = Any

int3 = tuple[int, int, int]
RecordDataFn = Callable[[h5py.File, int, int, Mapping], bool]


class DatasetOpts(TypedDict, total=False):
    # tracks kwargs for h5py.Group.create_dataset
    # -> keep in mind, TypedDict subclasses are only used to annotate regular dicts

    # The data type of the output datasets. Accepts most numpy types.
    dtype: np.dtype
    # kind of compression to use on the output data (if any)
    compression: str
    # Denotes compression settings (if any)
    compression_opts: str
    # Whether to use chunking and the chunk size
    chunks: Any


def dset_opts_from_args(args: argparse.Namespace) -> DatasetOpts:
    return {
        "dtype": args.dtype,
        "compression": args.compression_type,
        "compression_opts": args.compression_opts,
        "chunks": args.chunking,
    }


def to_int_triple(iterable: Iterable, *, min_val: Optional[int] = None) -> int3:
    # coerce the specified sequence to a 3 element tuple
    coerced_l = []
    for i, elem in enumerate(iterable):
        if i > 3:
            raise ValueError("invalid iterable for to_int_triple: more than 3 items")
        coerced = int(elem)
        if elem != coerced:
            raise ValueError(f"converting {elem} to 'int' changes the value")
        elif (min_val is not None) and (coerced < min_val):
            raise ValueError(f"min_val, {min_val}, exceeds int({elem})")
        coerced_l.append(coerced)
    if len(coerced_l) < 3:
        raise ValueError("invalid iterable for to_int_triple: less than 3 items")
    return tuple(coerced_l)


def _nprocs_and_blockcellshape(
    f: h5py.File,
    missing_nprocs_triple: Optional[int3] = None,
    f_is_concatenated: bool = False,
) -> tuple[int3, int3]:
    # first, infer nprocs
    f_has_nprocs = "nprocs" in f.attrs
    if missing_nprocs_triple is None and not f_has_nprocs:
        if "domain/blockid_location_arr" in f:
            return f["domain/blockid_location_arr"].shape
        raise ValueError("nprocs kwarg is required; it isn't an attr of f")
    elif missing_nprocs_triple is None:
        _nprocs = np.asarray(f.attrs["nprocs"])
    else:
        _nprocs = np.asarray(missing_nprocs_triple)
        if f_has_nprocs and np.any(_nprocs != np.asarray(f.attrs["nprocs"])):
            raise ValueError(
                "missing_nprocs_triple inconsistent with f.attrs['nprocs']"
            )
    nprocs = to_int_triple(_nprocs, min_val=1)

    # next, infer global dims
    if "dims" in f.attrs:
        global_dims = np.asarray(f.attrs["dims"])
    elif f_is_concatenated and "density" in f:
        global_dims = f["density"].shape
    else:
        RuntimeError("can't infer global dims")
    assert np.shape(global_dims) == (3,)

    # determine block_cell_shape
    _block_cell_shape, _remainder = np.divmod(global_dims, nprocs)
    assert np.all(_remainder == 0)

    return nprocs, to_int_triple(_block_cell_shape, min_val=1)


class BlockidLocationArrBuilder:
    """A blockid_location_arr builer. The array specifies each block's location

    Parameters
    ----------
    f
        source file to try to read values from
    nprocs
        must be provided if source file is missing the "nprocs" attribute
    """

    _arr: np.ndarray  # 3D array of i64
    block_cell_shape: int3

    def __init__(self, f: h5py.File, *, missing_nprocs_triple: Optional[int3] = None):
        nprocs, self.block_cell_shape = _nprocs_and_blockcellshape(
            f, missing_nprocs_triple=missing_nprocs_triple
        )
        try:
            self._arr = f["domain/blockid_location_arr"][...]
            assert nprocs == self._arr
        except KeyError:
            # Negative values denote unknown blockids
            self._arr = np.full(shape=nprocs, fill_value=-1, dtype="i8")

    @property
    def nprocs(self) -> int3:
        return self._arr.shape

    def final_arr(self) -> np.ndarray:  # return fully build array
        missing = np.sum(self._arr < 0)
        if missing > 0:
            raise RuntimeError(
                f"blockid_location_arr missing location info for {missing} blocks"
            )
        return self._arr

    def store_location(self, blockid: int, global_cell_offset: int3) -> Self:
        _idx, remainders = np.divmod(global_cell_offset, self.block_cell_shape)
        assert np.all(_idx >= 0) and np.all(remainders == 0)  # sanity-check
        idx = tuple(int(i) for i in _idx)

        if self._arr[idx] < 0:
            assert blockid not in self._arr  # sanity-check
            self._arr[idx] = blockid
        assert self._arr[idx] == blockid
        return self


def _record_field(
    dst_f: h5py.File,
    expected_store_block_count: int,
    stored_block_idx: int,
    data: Mapping,
    *,
    skip_fields: set,
    dset_opts: DatasetOpts,
) -> bool:
    """The core logic for recording fields to a file"""
    # get the field names to be copied
    names = [name for name in data.keys() if name not in skip_fields]

    # get the "field" group (create it if it doesn't exist yet)
    if "field" not in dst_f:
        field_grp = dst_f.create_group("field")
        for field_name, dset in data.items():
            assert len(dset.shape) == 3  # sanity check!
            shape = (expected_store_block_count,) + dset.shape
            field_grp.create_dataset(name=field_name, shape=shape, **dset_opts)
        created_grp = True
    else:
        field_grp = dst_f["field"]
        created_grp = False

    # now, store the field data
    assert len(names) == len(field_grp)  # sanity check!
    dest_sel = np.s_[stored_block_idx, ...]
    for name in names:
        field_grp[name].write_direct(data[name][...], dest_sel=dest_sel)

    return created_grp


def _configure_virtual_field(
    dst_f: h5py.File,
    blockid_location_arr: np.ndarray,
    stored_blockid_list: Sequence[int],
    dst_grp: str,
):
    """Create virtual datasets for each field. The virtual dataset acts like a
    3D array that spans the entire dataset.

    Parameters
    ----------
    dst_f
        Output file
    blockid_location_arr
        specifies the location of each block
    stored_blockid_list
        specifies the index of the blockid list
    dst_grp
        name of the group where we will write the datasets

    Notes
    -----
    This is only intended to limit breakage in existing scripts
    """
    total_block_count = np.prod(blockid_location_arr.shape)
    if len(stored_blockid_list) == total_block_count:
        blockid_to_idx_map = {
            blockid: idx for idx, blockid in enumerate(stored_blockid_list)
        }
    else:
        raise ValueError(
            "can't currently create virtual fields unless the file contains all blocks"
        )

    # calculate block-shape of a cell-centered field
    _cc_block_shape, _rem = np.divmod(dst_f.attrs["dims"], blockid_location_arr.shape)
    assert np.all(_rem == 0)  # sanity check!
    cc_block_shape = to_int_triple(_cc_block_shape)

    # expected shape for dataset storing cell-centered field
    src_dset_shape = (total_block_count,) + to_int_triple(cc_block_shape)
    cc_domain_shape = to_int_triple(
        np.multiply(blockid_location_arr.shape, cc_block_shape)
    )

    grp = dst_f.create_group(dst_grp)

    for field_name, field_dset in dst_f["field"].items():
        if field_dset.shape != src_dset_shape:
            raise ValueError(
                f"can't make a virtual dataset for {field} since the stored data "
                "doesn't have the expected shape for a cell-centered field"
            )
        vsrc = h5py.VirtualSource(field_dset)
        layout = h5py.VirtualLayout(
            shape=cc_domain_shape, dtype=field_dset.dtype, maxshape=cc_domain_shape
        )

        for idx3d, blockid in np.ndenumerate(blockid_location_arr):
            layout_slc = (
                slice(idx3d[0] * cc_block_shape[0], (idx3d[0] + 1) * cc_block_shape[0]),
                slice(idx3d[1] * cc_block_shape[1], (idx3d[1] + 1) * cc_block_shape[1]),
                slice(idx3d[2] * cc_block_shape[2], (idx3d[2] + 1) * cc_block_shape[2]),
            )
            layout[layout_slc] = vsrc[blockid_to_idx_map[blockid], :, :, :]
        grp.create_virtual_dataset(name=field_name, layout=layout)


class SnapBuilder:
    """A snapshot-file builder, providing fine control over the file's contents.

    The output file is organized according to the Hierarichal Schema. Users
    call builder methods to configure the output before using the `write`
    to produce the output. Logic is in place to ensure that the result isn't
    missing crucial information.

    Parameters
    ----------
    path
        Path to the output file.
    expected_store_block_count
        Specifies the number of blocks stored in the file. A value of ``None``
        (the default) indicates that all blocks will be stored.

    Notes
    -----
    We suggest using this type as a context manager, (i.e. in a ``with``
    statement), to help with proper cleanup if any errors occur.

    The various builder-methods are associated with the following aspects of the
    output file:

    - header-attribute-data: this must always be configured with `set_hdr`.
    - block-location-data: specifies the block location. This information is
      required by the builder. The info can be manually specified for a given
      block with `record_block_loc`. In practice, other methods can be used to
      implicitly specify this information
    - snapshot-data: this is the real core of the output. There are a few of
      "kinds" of snapshot data. Currently, we support the "field" kind. We plan
      to add support for "gravity" and "particle".

      - to configure the builder to record data of the specified "kind", call
        the ``{kind}_config``.
      - to actually record data for a given block, call ``{kind}_record``. The
        builder will report an error if you call this before ``{kind}_config``
        or before `set_hdr`.

    Methods
    -------
    write
        Write the output file
    cleanup
        Cleanup the builder (unnecessary if you call `write`)
    set_hdr
        Copy the header of an existing source file (must be called)
    record_block_loc
        Manually record a block's location
    field_config, particle_config
        Configure builder for recording field-data or particle-data
    field_record, particle_record
        Record the field-data or particle-data from a single block
    field_record_itr, particle_record_itr
        Record the field-data or particle-data from multiple blocks

    Examples
    --------
    The public methods all return ``self`` to allow chaining.

    Here's an example where we build a file with field data from entire domain
    >>> with SnapBuilder(out_path) as builder:  # open the snapshot-builder
    ...     builder.set_hdr(src_f) \
    ...         .field_config(opts=dset_opts, skip=skip_fields) \
    ...         .field_record_itr(itr) \
    ...         .write()

    In the near future, we plan to support code like the following snippets:
    >>> with SnapBuilder(out_path) as builder:  # open the snapshot-builder
    ...     builder.set_hdr(src_f) \
    ...         .particle_config(...) \
    ...         .particle_record_itr(itr) \
    ...         .write()
    
    or like:
    >>> with SnapBuilder(out_path) as builder:  # open the snapshot-builder
    ...     builder.set_hdr(src_f) \
    ...         .field_config(opts=dset_opts, skip=skip_fields) \
    ...         .field_record_itr(itr_field) \
    ...         .particle_config(...) \
    ...         .particle_record_itr(itr_particle) \
    ...         .write()
    """

    # note: the presence of type annotations means these are instance variables

    _tmp_final_path_pair: tuple[str, str]  # temporary and final paths
    _f: h5py.File  # File object that represents the file @ _tmp_final_path_pair[0]
    _expected_store_block_count: Optional[int]
    _recordpack_dict: dict[str, tuple[set, RecordDataFn]]
    _stored_blockid_list: list[int]
    _blockid_location_arr_builder: Optional[BlockidLocationArrBuilder] = None
    _write_virtual_field: Optional[bool] = None

    def __init__(
        self, path: os.PathLike, expected_store_block_count: Optional[int] = None
    ):
        if os.path.exists(path):
            raise OSError(errno.EEXIST, os.strerror(errno.EEXIST), path)
        self._tmp_final_path_pair = (f"{os.fsdecode(path)}-tmp", path)
        self._f = h5py.File(self._tmp_final_path_pair[0], "w-")
        self._expected_store_block_count = expected_store_block_count
        self._recordpack_dict = {}
        self._stored_blockid_list = []

    def _require(self, *attrs: str):  # common requirement-checking
        for a in attrs:
            if getattr(self, a) is not None:
                continue
            elif a == "_f":
                raise RuntimeError("already cleaned up the builder")
            elif a in ["_blockid_location_arr_builder", "_expected_store_block_count"]:
                raise RuntimeError("set_hdr method was never called")
            raise RuntimeError(f"{a} is invalid")

    def set_hdr(
        self,
        source_file: Union[os.PathLike, h5py.File],
        missing_nprocs_triple: Optional[int3] = None,
    ) -> Self:
        """Copy the header-attributes from source_file

        Parameters
        ----------
        source_file
            Used to set the header
        missing_nprocs_triple
            This must be provided if the information isn't part source_file

        Notes
        -----
        If source_file has the hierarichal-schema, all block location data will
        be read from source_file.
        """
        self._require("_f")
        if len(self._f.attrs) != 0:
            raise RuntimeError("It's an error to call set_hdr more than once")
        cm = nullcontext if isinstance(source_file, h5py.File) else h5py.File

        with cm(source_file) as src_f:
            # copy most of the header
            concat_internals.copy_header(src_f, self._f, skip_keys=["nprocs"])

            # construct the location-array-builder
            self._blockid_location_arr_builder = BlockidLocationArrBuilder(
                f=src_f, missing_nprocs_triple=missing_nprocs_triple
            )

        nprocs = np.array(self._blockid_location_arr_builder.nprocs)
        self._f.attrs["nprocs"] = nprocs
        assert "dims" in self._f.attrs  # this is a hard requirement

        tot_block_count = np.prod(nprocs)
        if self._expected_store_block_count is None:
            self._expected_store_block_count = tot_block_count
        elif self._expected_store_block_count > tot_block_count:
            raise RuntimeError(
                "builder is configured to store more blocks than actually exist"
            )
        return self

    def record_block_loc(self, blockid: int, global_cell_offset: int3) -> Self:
        """Manually record the specified block's location

        You usually don't need to manually call this method.

        Parameters
        ----------
        blockid
            The block's id
        global_cell_offset
            Location of the leftmost cell of the block in the conceptual grid
            spanning the entire domain
        """
        self._require("_f", "_blockid_location_arr_builder")
        self._blockid_location_arr_builder.store_location(blockid, global_cell_offset)
        return self

    def _setup_recorder(self, name: str, fn: Callable, **kwargs: Any):
        self._require("_f")
        if name in self._recordpack_dict:
            raise RuntimeError("the {name!r} recorder is already configured")
        self._recordpack_dict[name] = (set(), functools.partial(fn, **kwargs))

    def _record_data(
        self,
        blockid: int,
        data: Mapping,
        *,
        global_cell_offset: Optional[int3] = None,
        kind=None,
    ) -> Self:
        """Record the specified block's data of the specified kind

        Parameters
        ----------
        blockid
            The block's id
        data
            A dict-like object mapping field names to field data
        global_cell_offset
            Optionally specifies the location of the leftmost cell of the block
            in the conceptual grid spanning the entire domain. An error will be
            raised during the final ``write`` command if the locations are
            never specified.
        kind
            The data kind

        Notes
        -----
        This function is not intended to be called directly. Instead you should
        call methods like field_record or particle_record. (In these scenarios,
        you shouldn't specify the ``kind`` kwarg)
        """
        self._require("_f", "_expected_store_block_count")
        if kind not in self._recordpack_dict:
            raise RuntimeError(f"{kind}_config was never called")
        already_recorded_blockids, record_fn = self._recordpack_dict[kind]

        if blockid < 0:
            raise ValueError("blockid must not be nonnegative")
        elif blockid in already_recorded_blockids:
            raise RuntimeError(f"{kind}_record already called for blockid")

        # deal with the block's spatial location
        if hasattr(data, "attrs") and "offset" in data.attrs:
            _offset = to_int_triple(data.attrs["offset"])
            if global_cell_offset is not None:
                assert _offset == to_int_triple(global_cell_offset)
            self.record_block_loc(blockid, _offset)
        elif global_cell_offset is not None:
            self.record_block_loc(blockid, global_cell_offset)

        # handles case where `data` is a h5py.File for existing snapshot file
        data = data[kind] if kind in data else data

        # get the index in the file associated with blockid
        try:
            stored_block_idx = self._stored_blockid_list.index(blockid)
        except ValueError:
            stored_block_idx = len(self._stored_blockid_list)

        # store the data to the file
        record_fn(self._f, self._expected_store_block_count, stored_block_idx, data)

        # record that we've stored data for blockid
        already_recorded_blockids.add(blockid)
        if len(self._stored_blockid_list) == stored_block_idx:
            self._stored_blockid_list.append(blockid)

        return self

    def _record_data_itr(
        self, itr: Iterable, *, kind: str, file_paths: bool = False
    ) -> Self:
        """Records data (of the specified ``kind``) for multiple blocks

        iterable over blocks to write information of a given kind.

        Parameters
        ----------
        itr
            Iterable that returns ``(blockid, data)`` **OR**
            ``(blockid, data, global_cell_offset)``. When `file_paths`` is
            ``True``, ``data`` is treated as a file path.
        kind
            The data kind
        file_paths
            Controls the interpretation of ``itr``.

        Notes
        -----
        This function is not intended to be called directly. Instead you should
        call methods like field_record_itr or particle_record_itr. (In these
        scenarios, you shouldn't specify the ``kind`` kwarg)
        """

        cm = h5py.File if file_paths else nullcontext
        _packlen = None
        for pack in itr:
            if len(pack) != _packlen:
                if _packlen is not None:
                    raise ValueError("members of itr have inconsistent lengths")
                _packlen = len(pack)
                if _packlen < 2 or _packlen > 3:
                    raise ValueError("members of itr must hold 2 or 3 elements")
            blockid, data = pack[:2]
            global_cell_offset = pack[2] if _packlen == 3 else None

            with cm(data) as _data:
                self._record_data(
                    blockid=blockid,
                    data=_data,
                    global_cell_offset=global_cell_offset,
                    kind=kind,
                )
        return self

    field_record = functools.partialmethod(_record_data, kind="field")
    field_record_itr = functools.partialmethod(_record_data_itr, kind="field")

    def field_config(
        self,
        *,
        opts: Optional[DatasetOpts] = None,
        skip: Optional[Iterable] = None,
        legacy: bool = False,
    ) -> Self:
        """Configure the builder to write field-data

        Parameters
        ----------
        opts
            Optional kwargs for ``h5py.Group.create_dataset``.
        skip
            Fields to omit from output. Defaults to {}.
        legacy
            When True, the "field_legacy" HDF5 group will be created
        """
        opts = dict() if opts is None else opts
        skip = set() if skip is None else set(skip)
        self._setup_recorder(
            name="field", fn=_record_field, dset_opts=opts, skip_fields=skip
        )

        self._write_virtual_field = legacy
        return self

    particle_record = functools.partialmethod(_record_data, kind="particle")
    particle_record_itr = functools.partialmethod(_record_data_itr, kind="particle")

    def particle_config(
        self,
        store_particle_count: int,
        tot_particle_count: int,  # *, kwargs...
    ) -> Self:
        raise NotImplementedError("we need to implement this")
        # will look something like:
        # > self._setup_recorder(
        # >     name="particle",
        # >     fn=_record_particle,
        # >     store_particle_count=store_particle_count,
        # >     tot_particle_count=tot_particle_count,
        # >     kwargs...
        # > )
        # > return self

    def write(self):
        """finish writing the output file"""
        self._require(
            "_f", "_expected_store_block_count", "_blockid_location_arr_builder"
        )

        # generic sanity checks
        if "dims" not in self._f.attrs:
            raise RuntimeError("'dims' is missing from header")
        stored_block_count = len(self._stored_blockid_list)
        if stored_block_count == 0:
            raise RuntimeError("no snapshot-data is stored for any block")
        elif self._expected_store_block_count != stored_block_count:
            raise RuntimeError("snapshot-data is missing for some blocks")

        for kind, (recorded_set, _) in self._recordpack_dict.items():
            if len(recorded_set) != self._expected_store_block_count:
                raise RuntimeError(f"{kind}_record wasn't called enough times")

        # write the domain group of the file
        blockid_location_arr = self._blockid_location_arr_builder.final_arr()
        stored_blockid_list = np.array(self._stored_blockid_list)
        domain_grp = self._f.create_group("domain")
        domain_grp.create_dataset("blockid_location_arr", data=blockid_location_arr)
        domain_grp.create_dataset("stored_blockid_list", data=stored_blockid_list)

        if self._write_virtual_field:
            _configure_virtual_field(
                self._f, blockid_location_arr, stored_blockid_list, "field_legacy"
            )

        # save and close!
        self._f.close()
        self._f = None
        shutil.move(*self._tmp_final_path_pair)

    def cleanup(self):
        if self._f is not None:
            self._f.close()
            self._f = None
            os.remove(self._tmp_final_path_pair[0])

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs):
        self.cleanup()

    def __deepcopy__(self, memo):
        raise RuntimeError(f"can't deepcopy {self.__class__.__name__}")


##########################################
class _MockBlockFieldGroup(UserDict):
    # acts like a h5py.Group instance for a selected region. This is better
    # than using a dict of loaded fields since that uses a lot more memory

    def __init__(self, grp: Mapping, idx_map: Mapping):
        super().__init__(dict(grp.items()))
        self._idx_map = idx_map

    def __getitem__(self, key: str):
        return self.data[key][self._idx_map.get(key, ...)]

    def __repr__(self):
        return "<_MockBlockFieldGroup>"


def _product(*iterables: Iterable, advance_right: bool = True) -> Iterator:
    if advance_right:
        return itertools.product(*iterables)
    return map(lambda seq: seq[::-1], itertools.product(*iterables[::-1]))


def _iter_block_from_concat(
    src_f: h5py.File,
    missing_nprocs_triple: Optional[Sequence] = None,
) -> Iterable[tuple[int, _MockBlockFieldGroup, int3]]:
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

    if "field" in src_f:  # we dealing with file put into the new format
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
        nprocs, block_cell_shape = _nprocs_and_blockcellshape(
            f=src_f, missing_nprocs_triple=missing_nprocs_triple, f_is_concatenated=True
        )

        # At the time of writing, I'm pretty sure that advance_right=False should assign
        # blockids in a consistent manner to cholla
        blockid_offset_pairs = enumerate(
            _product(
                range(0, nprocs[0] * block_cell_shape[0], block_cell_shape[0]),
                range(0, nprocs[1] * block_cell_shape[1], block_cell_shape[1]),
                range(0, nprocs[2] * block_cell_shape[2], block_cell_shape[2]),
                advance_right=False,
            )
        )

        def _mk_idx(name, start):
            shape = [e for e in block_cell_shape]
            if name == "magnetic_x":
                shape[0] += 1
            elif name == "magnetic_y":
                shape[1] += 1
            elif name == "magnetic_z":
                shape[2] += 1
            return tuple(slice(start[i], start[i] + shape[i], 1) for i in range(3))

        for blockid, global_cell_offset in blockid_offset_pairs:
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
    legacy_field: bool = False,
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
    legacy_field
        When True, the "field_legacy" HDF5 group will be created.
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
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), src_path)
    elif not os.path.isdir(out_dir):
        raise OSError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), out_dir)
    elif os.path.exists(out_path):
        raise OSError(errno.EEXIST, os.strerror(errno.EEXIST), out_path)

    with h5py.File(src_path, "r") as src_f:  # open the source file
        # do some basic setup
        itr = _iter_block_from_concat(src_f, missing_nprocs_triple)

        with SnapBuilder(out_path) as builder:  # open the snapshot-builder
            builder.set_hdr(src_f, missing_nprocs_triple).field_config(
                opts=dset_opts, skip=skip_fields, legacy=legacy_field
            ).field_record_itr(itr).write()


parser = argparse.ArgumentParser(
    description="Repacks (previously concatenated) HDF5 snapshots for Cholla"
)
concat_internals.add_common_cli_args(parser, num_processes_choice="omit")
parser.add_argument(
    "--missing-nprocs-triple",
    nargs=3,
    default=None,
    type=int,
    help=(
        "This must be supplied when the 'nproc' attribute is missing from the "
        "input file. This specifies the number of processes along each axis."
    ),
)

parser.add_argument(
    "-s",
    "--src-path",
    required=True,
    help=(
        "template-path for the input file. This path should generally include "
        "{snap}, which will be replaced by the snapshot numbers"
    ),
)

concat_internals._add_legacyfield_arg(parser)


if __name__ == "__main__":
    args = parser.parse_args()

    repack_snapshot(
        out_dir=args.output_directory,
        src_path=args.src_path,
        missing_nprocs_triple=args.missing_nprocs_triple,
        skip_fields=args.skip_fields,
        dset_opts=dset_opts_from_args(args),
        legacy_field=args.legacy_field,
    )
