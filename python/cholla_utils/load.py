from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import os
import typing
import weakref

import h5py
import numpy as np

_IDX3D_TYPE = typing.Any

@dataclass(kw_only=True, slots=True)
class _BlockDiskMapping:
    """Contains info for mapping blockids to locations in hdf5 files"""

    # ``fname_template.format(blockid=...)`` produces the file containing
    # blockid (this can properly handle cases where all blocks are stored in a
    # single file)
    fname_template: str
    # group containing field data (empty string denotes the root group)
    field_group: str
    # maps blockid to an index that select all associated data from a
    # field-dataset
    field_idx_map: dict[int, tuple[int | slice, ...]]

def _infer_blockid_location_arr(fname_template, global_dims, arr_shape):
    # used when hdf5 files don't have an explicit "domain" group
    blockid_location_arr = np.empty(shape=tuple(int(e) for e in arr_shape), dtype="i8")
    if blockid_location_arr.size == 1:
        # primarily intended to handle the result of older concatenation scripts (it
        # also handles the case when only a single block is used, which is okay)
        blockid_location_arr[0, 0, 0] = 0
    else:  # handle distributed cholla datasets
        local_dims, rem = np.divmod(global_dims, blockid_location_arr.shape)
        assert np.all(rem == 0) and np.all(local_dims > 0)
        for blockid in range(0, blockid_location_arr.size):
            with h5py.File(fname_template.format(blockid=blockid), "r") as f:
                tmp, rem = np.divmod(f.attrs["offset"][:], local_dims)
            assert np.all(rem == 0)  # sanity check
            idx3D = tuple(int(e) for e in tmp)
            blockid_location_arr[idx3D] = blockid
    return blockid_location_arr


def _determine_data_layout(f: h5py.File) -> tuple[np.ndarray, _BlockDiskMapping]:
    """Determine the data layout of the snapshot

    The premise is that the basic different data formats shouldn't
    matter outside of this function.

    Note
    ----
    In principle, we could make stronger inferences about the ways that Cholla's
    output format is organized, when using distributed output files.
    """
    filename = f.filename

    # STEP 1: infer the template for all Cholla data-files by inspecting fname
    # ========================================================================
    # There are 2 conventions for the names of Cholla's data-files:
    #  1. "root.h5.{blockid}" is the standard format Cholla uses when writing
    #     files storing a single snapshot. Each MPI-rank will write a separate
    #     file & replace ``{blockid}`` with MPI-rank (Modern Cholla versions
    #     without MPI replace ``{blockid}`` with ``0``)
    #  2. "root.h5": is the standard format used by Cholla's concatenation
    #     scripts (older versions of Cholla without MPI also used this format
    #     to name outputs)
    _dir, _base = os.path.split(filename)
    _sep_i = _base.rfind(".")
    no_suffix = (
        (_sep_i == -1) or (_base[_sep_i:] == "") or (_base[:_sep_i] == "")
    )
    if no_suffix or not _base[_sep_i + 1 :].isdecimal():
        inferred_fname_template = filename  # filename doesn't change based on
                                            # blockid
        cur_filename_suffix = None
    else:
        inferred_fname_template = (
            os.path.join(_dir, _base[:_sep_i]) + ".{blockid}"
        )
        cur_filename_suffix = int(_base[_sep_i + 1 :])

    # STEP 2: Check whether the hdf5 file has a flat structure
    # ========================================================
    # Historically, we would always store datasets directly in the root group
    # of the data file. More recent concatenation scripts store no data in
    # groups.
    flat_structure = any(
        not isinstance(elem, h5py.Group) for elem in f.values()
    )

    # STEP 3: Extract basic domain info information from the file(s)
    # ==============================================================
    has_explicit_domain_info = "domain" in f
    if has_explicit_domain_info:
        # this branch primarily handles concatenated files made with newer logic
        blockid_location_arr = f["domain/blockid_location_arr"][...]
        field_idx_map = {
            blockid: (i, slice(None), slice(None), slice(None))
            for i, blockid in enumerate(f["domain/stored_blockid_list"][...])
        }
        consolidated_data = len(field_idx_map) == blockid_location_arr.size
        if not consolidated_data:
            # in the near future, we will support one of the 2 cases:
            # > if (flat_structure):
            # >     _common_idx = (slice(None), slice(None), slice(None))
            # > else:
            # >     _common_idx = (0, slice(None), slice(None), slice(None))
            # > field_idx_map = defaultdict(lambda arg=_common_idx: arg)
            raise ValueError(
                "no support for reading Cholla datasets where data is "
                "distributed among files that explicitly encode domain info."
            )
    else:  # (not has_explicit_domain_info)
        # this branch covers distributed datasets (directly written by Cholla)
        # and older concatenated files.
        #
        # historically, when the dataset is concatenated (in post-processing),
        # the "nprocs" hdf5 attribute has been dropped
        blockid_location_arr = _infer_blockid_location_arr(
            fname_template=inferred_fname_template,
            global_dims=f.attrs["dims"].astype("=i8"),
            arr_shape=f.attrs.get("nprocs", np.array([1, 1, 1])).astype("=i8"),
        )
        consolidated_data = blockid_location_arr.size == 1

        def _get_common_idx():
            return (slice(None), slice(None), slice(None))

        field_idx_map = defaultdict(_get_common_idx)

    # STEP 4: Finalize the fname template
    # ===================================
    if consolidated_data:
        fname_template = filename
    elif cur_filename_suffix != 0:
        raise ValueError(  # mostly just a sanity check!
            "filename passed to the load function for a distributed cholla "
            "dataset must end in '.0'"
        )
    else:
        fname_template = inferred_fname_template

    mapping = _BlockDiskMapping(
        fname_template=fname_template,
        field_group="" if flat_structure else "field",
        field_idx_map=field_idx_map,
    )
    return blockid_location_arr, mapping


class _FieldLoader:
    """Helper type that actually loads chunks of data."""

    blockid_location_arr: np.ndarray
    mapper: _BlockDiskMapping
    global_dims: [int, int, int]
    _cur_fname: str | None  # the currently open filename
    # _h5_cache is a list with 0 or 1 elements (it helps with _finalizer)
    _h5_cache: list[h5py.File]
    # weakref.finalize performs cleanup (more reliably than __del__)
    _finalizer: weakref.finalize

    def __init__(self, blockid_location_arr, mapper, global_dims):
        def _callback(thing_sequence):  # closes all things in thing_sequence
            for thing in thing_sequence:
                thing.close()

        self.blockid_location_arr = blockid_location_arr
        self.mapper = mapper
        self.global_dims = global_dims
        self._cur_fname = None
        self._h5_cache = []
        self._finalizer = weakref.finalize(self, _callback, self._h5_cache)


    def __enter__(self):
        """Treats self as a context manager"""
        return self


    def __exit__(self, *args):
        self.close()


    def close(self):
        self._finalizer()


    def _load_file(self, fname: str) -> h5py.File:
        has_loaded = len(self._h5_cache) == 1
        if self._cur_fname != fname or not has_loaded:
            if has_loaded:
                self._h5_cache.pop().close()
            self._h5_cache.append(h5py.File(fname,"r"))
        return self._h5_cache[0]


    def it_chunks(
        self, field_names: Sequence[str]
    ) -> Iterator[tuple[_IDX3D_TYPE, str, np.ndarray]]:
        """Returns an iterator that iterates over chunks of field data

        Yields
        ------
        global_idx
            Specifies the global set of indices corresponding to the output
            chunk
        field_name
            Specifies the field name that the chunk corresponds to
        chunk
            The actual chunk of data
        """
        dims_local = tuple(
            np.array(self.global_dims) // np.array(self.blockid_location_arr.shape)
        )

        for location_idx, blockid in np.ndenumerate(self.blockid_location_arr):
            # get the hdf5 group and index selection corresponding to blockid
            fname = self.mapper.fname_template.format(blockid=blockid)
            f = self._load_file(fname = fname)
            grp = f if self.mapper.field_group == "" else f[self.mapper.field_group]
            dataset_idx = self.mapper.field_idx_map[blockid]

            # determine the global indices that these correspond to
            tmp = []
            for i in range(len(location_idx)):
                start = dims_local[i] * location_idx[i]
                stop = dims_local[i] * (1 + location_idx[i])
                tmp.append(slice(start,stop))
            global_idx = tuple(tmp)

            # iterate over the specified field names
            for field_name in field_names:
                chunk = grp[field_name][dataset_idx].astype("f8")
                yield global_idx, field_name, chunk


@typing.overload
def load_field(snap: os.PathLike, /, field: str) -> np.ndarray:
    ...
@typing.overload
def load_field(snap: os.PathLike, /, field: Sequence[str]) -> dict[str, np.ndarray]:
    ...

def load_field(snap, /, field):
    """Loads in 1 or more fields for a snapshot

    Parameters
    ----------
    snap
        Path to the snapshot
    field
        The name of a single field to load or a sequence of field names
    """
    with h5py.File(snap, "r") as f:
        blockid_location_arr, mapper = _determine_data_layout(f)
        full_dims = tuple(int(e) for e in f.attrs["dims"][:].astype("=i8"))

    if isinstance(field, str):
        field_seq = [field]
    else:
        field_seq = field

    out = {field: np.empty(shape=full_dims, dtype='f8') for field in field_seq}
    with _FieldLoader(blockid_location_arr, mapper, full_dims) as field_loader:
        for out_idx, field_name, chunk in field_loader.it_chunks(field_seq):
            out[field_name][out_idx] = chunk

    if field != field_seq:
        return out[field]
    return out

def get_native_fields(snap: os.PathLike) -> Sequence[str]:
    """Returns the names of the fields that were saved to disk."""
    # this could be significantly more efficient
    with h5py.File(snap, "r") as f:
        _, mapper = _determine_data_layout(f)
        grp = f if mapper.field_group == "" else f[mapper.field_group]
        return tuple(grp.keys())
