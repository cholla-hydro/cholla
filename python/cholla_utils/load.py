from collections.abc import Mapping, Iterator, Sequence
from collections import defaultdict
from dataclasses import dataclass
import os
import typing
import weakref

import h5py
import numpy as np

from ._misc import (
    _DatasetDiskMapping,
    _determine_data_layout,
    _detect_particle_fields,
    ParticleType,
)

_IDX3D_TYPE = typing.Any

_FULL_REGION_SLICE_3D = tuple(slice(None) for _ in range(3))


def _format_field_idx(
    blockid: int, idx_map: Mapping[int, tuple[int | slice, ...]], idx: _IDX3D_TYPE
):
    """
    This function combines both
      1. `idx_map[block_id]`, which specifies the index selection to load
          all field values for the specified block from an hdf5 dataset as a
          3D array
      2. `idx`, which specifies the index selection to access field values in
          a region of interest from a 3D array (that initially holds all
          field values)

    The returned index tuple can be used to directly load values in the region
    of interest from the hdf5 dataset
    """
    full_block_idx = idx_map[blockid]
    ndim_full_block_idx = len(full_block_idx)
    if full_block_idx == _FULL_REGION_SLICE_3D:
        return idx
    elif (ndim_full_block_idx == 4) and full_block_idx[1:] == _FULL_REGION_SLICE_3D:
        return (full_block_idx[0], idx[0], idx[1], idx[2])
    elif (ndim_full_block_idx == 3) or (ndim_full_block_idx == 4):
        # this shouldn't happen with the current file organization schemes, but we
        # could try supporting this case
        raise NotImplementedError("can't handle this case (yet)")
    else:
        raise RuntimeError("there's probably a bug")


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


class _DataLoader:
    """Helper type that actually loads chunks of data."""

    blockid_location_arr: np.ndarray
    mapper: _DatasetDiskMapping
    global_dims: list[int, int, int]
    _cur_fname: str | None  # the currently open filename
    # _h5_cache is a list with 0 or 1 elements (it helps with _finalizer)
    _h5_cache: list[h5py.File]
    # weakref.finalize performs cleanup (more reliably than __del__)
    _finalizer: weakref.finalize

    def __init__(
        self,
        blockid_location_arr: np.ndarray,
        mapper: _DatasetDiskMapping,
        global_dims: list[int, int, int],
    ):
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
        idx_selector
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
            fname = self.mapper.field_mapping.fname_template.format(blockid=blockid)
            f = self._load_file(fname=fname)
            grp = f[self.mapper.field_mapping.h5_group_map["field"]]

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
                dataset_idx = _format_field_idx(
                    blockid, self.mapper.field_mapping.idx_map, load_idx
                )
                # iterate over the specified field names
                for field_name in field_names:
                    chunk = grp[field_name][dataset_idx].astype("f8")
                    yield out_idx, field_name, chunk

    def get_particle_counts(
        self, block_slice_triple: tuple[slice, slice, slice]
    ) -> dict[ParticleType, int]:
        mapper = self.mapper.particle_mapping
        if mapper is None:
            return {}

        # going to need to change this once we have multiple particle types
        assert len(self.mapper.particle_types) == 1

        fname_template = mapper.fname_template
        concatenated = fname_template.format(blockid=0) == fname_template
        itr = self.blockid_location_arr[block_slice_triple].flat

        if concatenated:
            counter = 0
            for blockid in itr:
                idx = mapper.idx_map[blockid]
                assert len(idx) == 1  # sanity check!
                slc = idx[0]
                assert (
                    (slc.start >= 0)
                    and (slc.stop >= 0)
                    and (slc.step is None or slc.step == 1)
                )  # another sanity check!
                counter += slc.stop - slc.start
        else:
            counter = 0
            for blockid in itr:
                # get the hdf5 group and index selection corresponding to blockid
                fname = mapper.fname_template.format(blockid=blockid)
                f = self._load_file(fname=fname)
                counter += f.attrs["n_particles_local"][0]
        return {self.mapper.particle_types[0]: int(counter)}

    def it_chunk_particle(
        self,
        ptype_prop_map: Mapping[ParticleType, Sequence[str]],
        block_slice_triple: tuple[slice, slice, slice],
    ) -> Iterator[tuple[slice, tuple[ParticleType, str], np.ndarray]]:
        """
        Returns an iterator that iterates over chunks of particle properties

        Parameters
        ----------
        ptype_prop_map
            Maps ptypes to the sequence of properties that we want to probe
        block_slice_triple
            slices the 3d regular array of blocks

        Yields
        ------
        out_idx: slice
            Specifies the set of indices of the concatenated output array that the
            yielded chunk corresponds to.
        tuple[str,str]
            Specifies (particle-type, property)
        chunk: np.ndarray
            The actual chunk of data
        """

        itr = self.blockid_location_arr[block_slice_triple].flat

        for ptype in ptype_prop_map:
            if ptype not in self.mapper.particle_types:
                raise ValueError(f"{ptype} is not a known particle type")

        n_loaded = {ptype: 0 for ptype in ptype_prop_map}
        mapper = self.mapper.particle_mapping
        for blockid in itr:
            # get the hdf5 group and index selection corresponding to blockid
            fname = mapper.fname_template.format(blockid=blockid)
            f = self._load_file(fname=fname)

            # get the indices in a generic dataset that correspond to blockid
            # (in the future, the indices probably need to be specific to both
            # the blockid and the particle-type)
            idx = mapper.idx_map[blockid]

            for ptype, props in ptype_prop_map.items():
                grp = f[self.mapper.particle_mapping.h5_group_map[ptype]]
                out_slc = None
                for prop in props:
                    data = grp[prop][idx]
                    if out_slc is None:
                        out_slc = slice(n_loaded[ptype], n_loaded[ptype] + data.size)
                    yield out_slc, (ptype, prop), data
                if out_slc is not None:
                    n_loaded[ptype] = out_slc.stop

    def get_particle_count(particle_type, block_slice_triple):
        pass


@typing.overload
def load_field(
    snap: os.PathLike, /, field: str, *, idx: tuple[slice | int, ...] | None
) -> np.ndarray: ...


@typing.overload
def load_field(
    snap: os.PathLike, /, field: Sequence[str], *, idx: tuple[slice | int, ...] | None
) -> dict[str, np.ndarray]: ...


def load_field(snap, /, field, *, idx=None):
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

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        A single array of field data, or a dictionary of field data (depending
        on whether the field is a single string or a sequence).
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
    with _DataLoader(blockid_location_arr, mapper, full_dims) as data_loader:
        itr = data_loader.it_chunks(field_seq, selector)
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
        grp = f[mapper.field_mapping.h5_group_map["field"]]
        return tuple(grp.keys())


def _coerce_block_idx(
    block_idx: Sequence[int | slice] | None, n_sim_dims: int
) -> tuple[slice, ...]:
    if block_idx is None:
        return tuple(slice(None) for _ in range(n_sim_dims))

    try:
        len_block_idx = len(block_idx)
    except TypeError:
        raise TypeError("block_idx must be coercible to a tuple") from None

    if len_block_idx != n_sim_dims:
        raise ValueError(
            "when specified, block_idx must be a tuple with the same number "
            "of elements as the simulation has dimensions"
        )
    elif any(not isinstance(e, int | slice) for e in block_idx):
        raise TypeError("all elements of block_idx must ints or slices")

    def _to_slc(arg):
        return slice(-1, None) if arg == -1 else slice(arg, arg + 1)

    return tuple(e if isinstance(e, slice) else _to_slc(e) for e in block_idx)


@typing.overload
def load_particle(
    snap: os.PathLike,
    /,
    ptype_prop_pair: tuple[[ParticleType, str]],
    *,
    block_idx: Sequence[int | slice] | None,
) -> np.ndarray: ...


@typing.overload
def load_particle(
    snap: os.PathLike,
    /,
    ptype_prop_pair: tuple[ParticleType, str],
    *,
    block_idx: Sequence[int | slice] | None,
) -> dict[tuple[ParticleType, str], np.ndarray]: ...


def load_particle(snap, /, ptype_prop_pair, *, block_idx=None):
    """Loads in 1 or more fields for a snapshot

    Parameters
    ----------
    snap
        Path to the snapshot
    ptype_prop_pair
        A single `(<particle-type>, <particle-property>)` pair or a sequence of pairs.
    block_idx
        Optionally specifies a tuple of 3 `slice` instances that specify the
        subset of blocks to load from disk. The easiest way to specify this
        is by using numpy.s_[...].

    Returns
    -------
    np.ndarray or dict[tuple[str, str], np.ndarray]
        A single 1D array of particle properties, or a dictionary of particle
        properties (depending on whether `ptype_prop_pair` is a single pair, or a
        sequence).
    """
    with h5py.File(snap, "r") as f:
        blockid_location_arr, mapper = _determine_data_layout(f)
        full_dims = tuple(int(e) for e in f.attrs["dims"][:].astype("=i8"))

    # construct ptype_prop_map
    if (
        isinstance(ptype_prop_pair, tuple)
        and (len(ptype_prop_pair) == 2)
        and isinstance(ptype_prop_pair[0], str)
        and isinstance(ptype_prop_pair[1], str)
    ):
        return_dict = False
        ptype_prop_map = {ptype_prop_pair[0]: [ptype_prop_pair[1]]}
    else:
        return_dict = True
        ptype_prop_map = defaultdict(list)
        for ptype, prop in ptype_prop_pair:
            ptype_prop_map[ptype].append(prop)

    # coerce the block_idx argument
    block_idx = _coerce_block_idx(block_idx, len(full_dims))

    with _DataLoader(blockid_location_arr, mapper, full_dims) as data_loader:
        # get the total number of particles that we will read
        particle_counts = data_loader.get_particle_counts(block_idx)

        # (this is a much sillier way of getting particle_counts from before)
        # _simpler_map = {ptype: props[:1] for ptype, props in ptype_prop_map.items()}
        # alt_particle_counts = {ptype: 0 for ptype in ptype_prop_map}
        # itr = data_loader.it_chunk_particle(_simpler_map, block_idx)
        # for out_idx, ptype_prop_pair, chunk in itr:
        #     alt_particle_counts[ptype_prop_pair[0]] = out_idx.stop
        # assert alt_particle_counts == particle_counts

        itr = data_loader.it_chunk_particle(ptype_prop_map, block_idx)
        particle_data = {}
        for out_idx, ptype_prop_pair, chunk in itr:
            # allocated the buffer the first time we encounter a ptype_prop_pair. We
            # wait to allocate since properties (namely particle_IDs) may not be floats
            if ptype_prop_pair not in particle_data:
                shape = (particle_counts[ptype_prop_pair[0]],)
                particle_data[ptype_prop_pair] = np.empty(shape, dtype=chunk.dtype)
            particle_data[ptype_prop_pair][out_idx] = chunk

    if return_dict:
        return particle_data
    return next(iter(particle_data.values()))


def get_native_ptype_properties(snap: os.PathLike) -> Sequence[[ParticleType, str]]:
    """
    Returns the 2-tuples of (ptype, property) for each property that of
    each particle-type that was saved to disk.
    """
    # this could be significantly more efficient
    with h5py.File(snap, "r") as f:
        _, mapper = _determine_data_layout(f)
        return _detect_particle_fields(mapper)


def get_native_root_attributes(snap: os.PathLike) -> dict:
    """Returns the raw root attributes"""
    # I'm not sure I really like this function, but at the same time, I don't think
    # we want to commit to any particular approach of digesting and organizing header
    # information yet
    with h5py.File(snap, "r") as f:
        return dict(f.attrs)
