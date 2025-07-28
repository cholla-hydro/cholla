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
    no_suffix = (_sep_i == -1) or (_base[_sep_i:] == "") or (_base[:_sep_i] == "")
    if no_suffix or not _base[_sep_i + 1 :].isdecimal():
        # filename doesn't change based on blockid
        inferred_fname_template = filename
        cur_filename_suffix = None
    else:
        inferred_fname_template = os.path.join(_dir, _base[:_sep_i]) + ".{blockid}"
        cur_filename_suffix = int(_base[_sep_i + 1 :])

    # STEP 2: Check whether the hdf5 file has a flat structure
    # ========================================================
    # Historically, we would always store datasets directly in the root group
    # of the data file. More recent concatenation scripts store no data in
    # groups.
    flat_structure = any(not isinstance(elem, h5py.Group) for elem in f.values())

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


@dataclass(kw_only=True, slots=True)
class _GlobalIdx3DSelection:
    """Helper type encoding the continuous region of a dataset to load.

    This region is specified as indices in a particular dataset's
    global index space. Importantly, this type can only represent
    cuboid regions that can also be specified in terms of a slice along
    each axis, where the step size is 1.

    To better illustrate the purpose of this type, imagine that we
    wanted to load in a subregion of a density field from a simulation.
    - The simple, inefficient way to do this would be to:
      1. load in the entire concatenated density field into a single
         numpy array. For the sake of argument, lets imagine that we
         store that array in a variable in ``concat_density``.
      2. Then we use numpy slicing expressions to create a view on the
         subregion of interest. If the output array is called
         ``density``, the then we would write code of the form:
         ``denisty = concat_density[<IndexExpression>]``. More concrete
         examples include:
         - ``denisty = concat_density[3:6, :, -4:-3]``
         - ``density = concat_density[:, 100:200, 4]``
    - This type encodes the contents of ``<IndexExpression>``. This
      type is constrained to it only represents slices along an axis
      where the step size is exactly ``1``. In other words,
      - slice expressions like ``:``, ``:4``, ``-1:``, ``1:-3``,
        ``-3:512:1``, etc. can all be represented.
      - slice expressions like like `::2`, `-7:-30:-1`, etc. can't be
        represented

    Note
    ----
    An earlier implementation tried to pass directly pass around a
    slice tuple, rather than defining this type, but that made the code
    much too complicated. In other words, this type mostly exists to
    simplify bookkeeping and code organization
    """

    # primary representation of the selection region. Importantly, we
    # always use nonnegative integers for encoding start and stop
    slice_triple: tuple[slice, slice, slice]

    # indicates whether the index was originally specified as an int.
    # This information may be used to coerce 3D arrays down to the
    # appropriate shape
    originally_int: tuple[bool, bool, bool]

    # shape of 3D array to hold selected region encoded by `self`
    output_arr_3D_shape: tuple[int, int, int]

    def __init__(self, *, idx, full_dims: Sequence[int]):
        """
        Parameters
        ----------
        idx
            Is an index tuple encoding the selection region. The
            easiest way to specify this is to invoke
            ``numpy.s_[<IndexExpression]`` or
            ``numpy.index_exp[<IndexExpression>]``.
        full_dim
            Specifies the shape of a cell-centered field after the data
            has been concatenated
        """

        # Step 1 coerce to slice_triple and originally_int
        if len(full_dims) != 3:
            raise ValueError("full_dim must hold 3 values")
        elif idx is None:
            slice_triple = [slice(0, axlen, 1) for axlen in full_dims]
            originally_int = [False for _ in range(3)]
        elif len(idx) != 3:
            raise ValueError("when idx is specified, in must have 3 components")
        else:
            slice_triple = []
            originally_int = []
            for i, comp in enumerate(idx):
                dim_len = full_dims[i]
                is_int = isinstance(comp, int)
                is_slc = isinstance(comp, slice)
                if is_int and ((comp < -dim_len) or (comp >= dim_len)):
                    raise ValueError(
                        f"idx[{i}] doesn't lie within the global index space"
                    )
                elif is_int:
                    comp = comp if comp >= 0 else comp + dim_len
                    slice_triple.append(slice(comp, comp + 1))
                    originally_int.append(True)
                elif is_slc and (comp.step is not None and comp.step != 1):
                    raise ValueError("At this time, we only support a slice step of 1")
                elif is_slc:
                    if comp.start is None:
                        start = 0
                    elif comp.start < -dim_len or comp.start >= dim_len:
                        raise ValueError(
                            f"slice at idx[{i}] starts outside the global index space"
                        )
                    else:
                        start = comp.start if comp.start >= 0 else comp.start + dim_len

                    if comp.stop is None:
                        stop = dim_len
                    elif comp.stop <= -dim_len or comp.stop > dim_len:
                        raise ValueError(
                            f"slice at idx[{i}] has an invalid stopping value"
                        )
                    else:
                        stop = comp.stop if comp.stop >= 0 else comp.stop + dim_len

                    if stop <= start:  # we verified that step is always 1
                        raise ValueError(f"slice at idx[{i}] selects no indices")
                    else:
                        slice_triple.append(slice(start, stop, 1))
                        originally_int.append(False)
                else:
                    raise TypeError(
                        f"the type of idx[{i}], {idx[i].__class__.__name__}, isn't "
                        "currently supported"
                    )
        # Step 2: record attributes
        # (we use object.__setattr__ since the type is is immutable)
        object.__setattr__(self, "slice_triple", tuple(slice_triple))
        object.__setattr__(self, "originally_int", tuple(originally_int))
        output_arr_3D_shape = tuple(slc.stop - slc.start for slc in self.slice_triple)
        object.__setattr__(self, "output_arr_3D_shape", output_arr_3D_shape)

    def massage_final_shape(self, arr):
        # returns a view of arr (i.e. no copy) that adjusts the dimensionality of arr
        # based upon whether the index-expressions included integers
        tmp_idx = tuple(0 if e else slice(None) for e in self.originally_int)
        return arr[tmp_idx]

    def infer_block_idx_pair(self, block_global_idx_bounds: tuple[slice, slice, slice]):
        """
        Infer index tuple pair for a simulation block's data in the selection region
        represented by self

        Parameters
        ----------
        block_global_idx_bounds
            Specifies the start and stopping values, in the global concatenated index
            space, that describes a particular simulation block

        Returns
        -------
        load_idx
            When none of the simulation block's data lies with the selection region,
            this is ``None``. Otherwise, this holds a tuple of 3 slices that to be used
            to select the subset of the simulation block's data that lies within the
            selection region
        out_idx
            When none of the simulation block's data lies with the selection region,
            this is ``None``. Otherwise, this holds a tuple of 3 slices that to be used
            to specify where in the output array (used to store the concatenated
            selection region) the data loaded with ``load_idx`` should be stored.
        """
        load_idx, out_idx = [], []
        for i in range(3):
            sel_bound_start = self.slice_triple[i].start
            sel_bound_stop = self.slice_triple[i].stop

            block_bound_start = block_global_idx_bounds[i].start
            block_bound_stop = block_global_idx_bounds[i].stop

            # let's infer the bounds on the region of overlap (in global index space)
            ov_start = max(block_bound_start, sel_bound_start)
            ov_stop = min(block_bound_stop, sel_bound_stop)

            if (
                (ov_start >= block_bound_stop)
                or (ov_stop <= block_bound_start)
                or (ov_start >= sel_bound_stop)
                or (ov_stop <= sel_bound_start)
            ):
                return None, None  # don't load any data from this simulation block
            load_idx.append(
                slice(ov_start - block_bound_start, ov_stop - block_bound_start)
            )
            out_idx.append(slice(ov_start - sel_bound_start, ov_stop - sel_bound_start))

            # sanity checks:
            length = ov_stop - ov_start
            assert (load_idx[i].stop - load_idx[i].start) == length
            assert (out_idx[i].stop - out_idx[i].start) == length
        return tuple(load_idx), tuple(out_idx)


class _FieldLoader:
    """Helper type that actually loads chunks of data."""

    blockid_location_arr: np.ndarray
    mapper: _BlockDiskMapping
    global_dims: list[int, int, int]
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
            self._h5_cache.append(h5py.File(fname, "r"))
        return self._h5_cache[0]

    def it_chunks(
        self, field_names: Sequence[str], idx_selector: _GlobalIdx3DSelection
    ) -> Iterator[tuple[_IDX3D_TYPE, str, np.ndarray]]:
        """Returns an iterator that iterates over chunks of field data

        Parameters
        ----------
        field_names
            The names of the fields that will be loaded
        slice_triple
            Represents information about the regions that we want to select.

        Yields
        ------
        out_idx
            Specifies the set of indices of the concatenated output array that the
            yielded chunk corresponds to.
        field_name
            Specifies the field name that the chunk corresponds to
        chunk
            The actual chunk of data
        """
        dims_local = tuple(
            np.array(self.global_dims) // np.array(self.blockid_location_arr.shape)
        )

        # TODO: use information encoded within idx_selector to make an iterator that
        #       only includes essential data chunks (this could save a little time if
        #       we wanted load a very small amount of data from a dataset distributed
        #       across many files
        itr = np.ndenumerate(self.blockid_location_arr)

        for location_idx, blockid in itr:
            # get the hdf5 group and index selection corresponding to blockid
            fname = self.mapper.fname_template.format(blockid=blockid)
            f = self._load_file(fname=fname)
            grp = f if self.mapper.field_group == "" else f[self.mapper.field_group]
            dataset_idx = self.mapper.field_idx_map[blockid]

            # determine the global indices that the chunk corresponds to
            tmp = []
            for i in range(len(location_idx)):
                start = int(dims_local[i] * location_idx[i])
                stop = int(dims_local[i] * (1 + location_idx[i]))
                tmp.append(slice(start, stop))
            block_global_bounds = tuple(tmp)

            load_idx, out_idx = idx_selector.infer_block_idx_pair(block_global_bounds)
            if load_idx is None:
                continue
            else:
                # iterate over the specified field names
                for field_name in field_names:
                    # TODO: it would be nice to merge dataset_idx and load_idx together
                    #       so that we load as little data as possible
                    chunk = grp[field_name][dataset_idx][load_idx].astype("f8")
                    yield out_idx, field_name, chunk


@typing.overload
def load_field(
    snap: os.PathLike, /, field: str, idx: tuple[slice | int, ...] | None
) -> np.ndarray: ...


@typing.overload
def load_field(
    snap: os.PathLike, /, field: Sequence[str], idx: tuple[slice | int, ...] | None
) -> dict[str, np.ndarray]: ...


def load_field(snap, /, field, idx=None):
    """Loads in 1 or more fields for a snapshot

    Parameters
    ----------
    snap
        Path to the snapshot
    field
        The name of a single field to load or a sequence of field names
    idx
        Optionally specifies a tuple of 3 `slice` instances that specify the
        subset of indices to load from disk. The easiest way to specify this
        is by using numpy.s_[...]
    """
    with h5py.File(snap, "r") as f:
        blockid_location_arr, mapper = _determine_data_layout(f)
        full_dims = tuple(int(e) for e in f.attrs["dims"][:].astype("=i8"))

    if isinstance(field, str):
        field_seq = [field]
    else:
        field_seq = field

    # here we coerce the idx argument
    selector = _GlobalIdx3DSelection(idx=idx, full_dims=full_dims)

    # allocate the output buffers
    nominal_shape = selector.output_arr_3D_shape
    field_dict = {f: np.empty(shape=nominal_shape, dtype="f8") for f in field_seq}

    # fill the output buffers
    with _FieldLoader(blockid_location_arr, mapper, full_dims) as field_loader:
        itr = field_loader.it_chunks(field_seq, selector)
        for out_idx, field_name, chunk in itr:
            field_dict[field_name][out_idx] = chunk

    # massage the shape of the output buffers (it adjusts dimensionality if any entry
    # in `idx` was an integer
    for field_name in field_seq:
        field_dict[field_name] = selector.massage_final_shape(field_dict[field_name])

    if field != field_seq:
        return field_dict[field]
    return field_dict


def get_native_fields(snap: os.PathLike) -> Sequence[str]:
    """Returns the names of the fields that were saved to disk."""
    # this could be significantly more efficient
    with h5py.File(snap, "r") as f:
        _, mapper = _determine_data_layout(f)
        grp = f if mapper.field_group == "" else f[mapper.field_group]
        return tuple(grp.keys())


def get_native_root_attributes(snap: os.PathLike) -> dict:
    """Returns the raw root attributes"""
    # I'm not sure I really like this function, but at the same time, I don't think
    # we want to commit to any particular approach of digesting and organizing header
    # information yet
    with h5py.File(snap, "r") as f:
        return dict(f.attrs)
