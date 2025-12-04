import enum
import os
import logging
import typing
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import h5py
import numpy as np

# DataType might be "field" or the name of a particular particle type
ParticleType: typing.TypeAlias = str
DataType: typing.TypeAlias = typing.Union[str, ParticleType]

logger = logging.getLogger("cholla_utils")
logger.setLevel(logging.DEBUG)
mylog = logger

class _CachedH5Openner:
    """
    A simple context manager that helps implement the idiom where data is read
    from (or written to) one or more HDF5 and we want to wait to close the
    previous HDF5 file until it is time to open a new file. This lets us avoid
    overhead in cases where we would close and then immediately reopen the
    same file.

    By using a context manager, we're able to properly cleanup in the event
    that an exception occurs.
    """

    def __init__(self, mode="r"):
        self._filename = None
        self._fh = None
        self._mode = mode

    def open_fh(self, filename):
        if self._filename == filename:
            return self._fh
        if self._fh is not None:
            self._fh.close()
        self._fh = h5py.File(filename, self._mode)
        self._filename = filename
        return self._fh

    def __enter__(self):
        return self

    def __exit__(self, exc, value, tb):
        if self._fh is not None:
            self._fh.close()


class ChollaDataFmt(enum.Enum):
    """Describes the format of the grid data"""

    # the format directly written by Cholla (each block is written to a separate file)
    DISTRIBUTED = (enum.auto(), False)
    # Cholla's older concatenation scripts (that are no longer available), would
    # combine all blocks into 1 giant block. The resulting generally appears as if
    # Cholla was run with a single process that evolved a single giant block of data
    LEGACY_CONCAT = (enum.auto(), True)
    # Cholla's newer concatenation scripts combine all of the data into a single file,
    # but retains the original block structure
    CONCAT = (enum.auto(), True)

    def __new__(cls, value: typing.Any, is_single_file: bool):
        # based on example from docs
        if isinstance(value, enum.auto):
            value = len(cls.__members__) + 1

        obj = object.__new__(cls)
        obj._value_ = value
        obj.is_single_file = is_single_file
        return obj

    def __repr__(self):
        # based on example from docs (when we want to hide the underlying value)
        return f"<{self.__class__.__name__}, {self.name}>"


@dataclass(kw_only=True, slots=True, frozen=True)
class _BlockDiskMapping:
    """
    Contains info for mapping blockids to locations in one or more hdf5 files.

    Depending on the context, an instance could be used to track the disk
    location for mesh field data or particle data from a Cholla simulation.
    """

    # ``fname_template.format(blockid=...)`` produces the file containing blockid (this
    # can properly handle cases where all blocks are stored in a single file)
    fname_template: str
    # ``h5_group_map[data_type]`` produces the hdf5 group containing relevant data.
    h5_group_map: dict[DataType, str]
    # maps blockid to an index that select all associated data from a field-dataset
    idx_map: Mapping[int, tuple[int | slice, ...]]


@dataclass(kw_only=True, slots=True, frozen=True)
class _DatasetDiskMapping:
    """Contains info for locating field-data and particle-data in hdf5 files"""

    field_mapping: _BlockDiskMapping
    particle_mapping: _BlockDiskMapping | None
    particle_types: tuple[ParticleType, ...]


def _infer_particle_mapping_and_types(
    block0_fluid_fname: str, fluid_data_fmt: ChollaDataFmt
) -> tuple[_BlockDiskMapping | None, tuple[ParticleType, ...]]:
    """
    Try to infer the how particle data is organized on disk.

    Parameters
    ----------
    block0_fluid_fname: str
        Specifies the path to the file containing fluid for blockid 0. (This
        should always be the same as the file that was passed to
        ``cholla_utils.load_field``).
    fluid_data_fmt: ChollaDataFmt
        The format of the fluid data in the Cholla dataset.

    Returns
    -------
    particle_mapping: _BlockDiskMapping or None
        This will be None if the template can't be inferred or there aren't
        any files at the expected locations.
    particle_types: tuple of strings
        Specify the kinds of particles included in the dataset
    """
    # we try to be very explicit why we end/skip the search for particle data
    # (to be transparent to end-users)
    match fluid_data_fmt:
        case ChollaDataFmt.LEGACY_CONCAT:
            mylog.info(
                "Skipping check for particle-data when reading data with Cholla's "
                "legacy concatenation format"
            )
            # the fundamental problem stems from the way that Cholla's legacy fluid
            # concatenation script combines all blocks into 1 giant block.
            # - technically, Cholla had legacy concatenation scripts that did the same
            #   thing for particle-data. But, to my knowledge nobody ever used those
            #   scripts for particle datasets (so we don't support it)
            # - The only supported way of loading particle-data retains the block
            #   structure. While we could support this mismatch of particle and fluid
            #   data, it would involve a lot of work (and I don't think it will ever
            #   actually come up in the real world)
            return None, ()
        case ChollaDataFmt.CONCAT:
            expected_suffix = ".h5"
        case ChollaDataFmt.DISTRIBUTED:
            expected_suffix = ".h5.0"
        case _:
            raise RuntimeError("should be unreachable")

    suf_len = len(expected_suffix)
    min_basename_len = suf_len + 1

    if not block0_fluid_fname.endswith(expected_suffix):
        mylog.info(
            "Skip check for particle-data: the path to the fluid data file "
            "(containing data for blockid 0) doesn't have the expected suffix, "
            f"{expected_suffix!r} (for the {fluid_data_fmt.name} format)"
        )
        return None, ()
    elif (
        (len(block0_fluid_fname) < min_basename_len)
        or (block0_fluid_fname[-min_basename_len] == os.sep)
        or (block0_fluid_fname[-min_basename_len] == os.altsep)
    ):
        mylog.info(
            "Skip check for particle-data: the basename of the fluid data file "
            f"doesn't contain any characters before the {expected_suffix!r} suffix"
        )
        return None, ()
    fname_template = f"{block0_fluid_fname[:-suf_len]}_particles.h5.{{blockid:d}}"
    concat_fname = f"{block0_fluid_fname[:-suf_len]}_particles.h5"

    if os.path.isfile(fname_template.format(blockid=0)):
        ptypes = ("io",)
        particle_mapping = _BlockDiskMapping(
            fname_template=fname_template,
            h5_group_map={"io": "./"},
            idx_map=defaultdict(lambda: (slice(None),)),
        )
    elif os.path.isfile(concat_fname):
        with h5py.File(concat_fname, "r") as f:
            ptypes = tuple(f["particle"].keys())
            assert len(ptypes) == 1  # temporary sanity check!
            idx_map = {}
            stop_block_idx_slc = f["particle"][ptypes[0]]["stop_block_idx_slc"][()]
            for stored_idx, blockid in enumerate(f["domain/stored_blockid_list"][()]):
                if stored_idx == 0:
                    start = 0
                else:
                    start = stop_block_idx_slc[stored_idx - 1]
                idx_map[blockid] = (slice(start, stop_block_idx_slc[stored_idx]),)
        h5_group_map = {ptype: f"particle/{ptype}" for ptype in ptypes}
        particle_mapping = _BlockDiskMapping(
            fname_template=concat_fname, h5_group_map=h5_group_map, idx_map=idx_map
        )

    else:
        mylog.info("No particle data was found")
        particle_mapping = None
        ptypes = ()

    return particle_mapping, ptypes


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


def _determine_data_layout(f: h5py.File) -> tuple[np.ndarray, _DatasetDiskMapping]:
    """Determine the data layout of the snapshot

    The premise is that the basic different data formats shouldn't
    matter outside of this function."""
    filename = os.fsdecode(f.filename)

    # STEP 1: infer the template for all Cholla data-files by inspecting filename
    # ===========================================================================
    # There are 2 conventions for the names of Cholla's data-files:
    #  1. "root.h5.{blockid}" is the standard format Cholla uses when writing files
    #     storing a single snapshot. Each MPI-rank will write a separate file and
    #     replace ``{blockid}`` with MPI-rank (Modern Cholla versions without MPI
    #     replace ``{blockid}`` with ``0``)
    #  2. "root.h5": is the standard format used by Cholla's concatenation scripts
    #     (older versions of Cholla without MPI also used this format to name outputs)
    inferred_fname_template, cur_filename_suffix = _infer_fname_template(filename)

    # STEP 2: Check whether the hdf5 file has a flat structure
    # ========================================================
    # Historically, we would always store datasets directly in the root group of the
    # data file. More recent concatenation scripts store no data in groups.
    flat_structure = any(not isinstance(elem, h5py.Group) for elem in f.values())

    # STEP 3: Extract basic domain info information from the file(s)
    # ==============================================================
    has_explicit_domain_info = "domain" in f
    if has_explicit_domain_info:
        # this branch primarily handles concatenated files made with newer logic
        blockid_location_arr = f["domain/blockid_location_arr"][...]
        field_idx_map = {
            int(blockid): (i, slice(None), slice(None), slice(None))
            for i, blockid in enumerate(f["domain/stored_blockid_list"][...])
        }
        if len(field_idx_map) == blockid_location_arr.size:
            data_fmt = ChollaDataFmt.CONCAT
        else:
            # in the near future, we may support one of the 2 cases:
            # > if (flat_structure):
            # >     _common_idx = (slice(None), slice(None), slice(None))
            # > else:
            # >     _common_idx = (0, slice(None), slice(None), slice(None))
            # > field_idx_map = defaultdict(lambda arg=_common_idx: arg)
            raise ValueError(
                "no support for reading Cholla datasets where data is distributed "
                "among files that explicitly encode domain info."
            )
    else:  # (not has_explicit_domain_info)
        # this branch covers distributed datasets (directly written by Cholla) and
        # older concatenated files.
        #
        # historically, when the dataset is concatenated (in post-processing),
        # the "nprocs" hdf5 attribute has been dropped
        blockid_location_arr = _infer_blockid_location_arr(
            fname_template=inferred_fname_template,
            global_dims=f.attrs["dims"].astype("=i8"),
            arr_shape=f.attrs.get("nprocs", np.array([1, 1, 1])).astype("=i8"),
        )
        if blockid_location_arr.size == 1:
            data_fmt = ChollaDataFmt.LEGACY_CONCAT
        else:
            data_fmt = ChollaDataFmt.DISTRIBUTED

        def _get_common_idx():
            return (slice(None), slice(None), slice(None))

        field_idx_map = defaultdict(_get_common_idx)

    # STEP 4: Finalize the fname template
    # ===================================
    match data_fmt:
        case ChollaDataFmt.LEGACY_CONCAT | ChollaDataFmt.CONCAT:
            fname_template = filename
        case ChollaDataFmt.DISTRIBUTED:
            if cur_filename_suffix != 0:
                raise ValueError(  # mostly just a sanity check!
                    "filename passed to cholla_utils.load for a distributed "
                    "cholla dataset must end in '.0'"
                )
            fname_template = inferred_fname_template
        case _:
            raise RuntimeError("should be unreachable")
    field_mapping = _BlockDiskMapping(
        fname_template=fname_template,
        h5_group_map={"cholla": "./" if flat_structure else "field"},
        idx_map=field_idx_map,
    )

    # STEP 5: Check if there is a particle dataset
    # ============================================
    particle_mapping, particle_types = _infer_particle_mapping_and_types(
        block0_fluid_fname=filename, fluid_data_fmt=data_fmt
    )

    dset_mapping = _DatasetDiskMapping(
        field_mapping=field_mapping,
        particle_mapping=particle_mapping,
        particle_types=particle_types,
    )
    return blockid_location_arr, dset_mapping


def _infer_fname_template(filename: str) -> tuple[str, int | None]:
    """Infers the template for all Cholla data-files based on the filename
    passed to ``cholla_utils.load``.

    string from the process-id suffix, and returns both parts in a 2-tuple.

    There are 2 conventions for the names of Cholla's data-files:
      1. "root.h5.{blockid}" is the standard format Cholla uses when writing
         files storing a single snapshot. Each MPI-rank will write a separate
         file and replace ``{blockid}`` with MPI-rank (Modern Cholla versions
         without MPI replace ``{blockid}`` with ``0``)
      2. "root.h5": is the standard format used by Cholla's concatenation
         scripts (older versions of Cholla without MPI also used this format
         to name outputs)

    Returns
    -------
    template: str
        The path to the file containing a blockid is given by calling
        ``template.format(blockid=<blockid>)``. (This will work whether
        all blocks are stored in 1 file or blocks are distributed across
        files)
    cur_blockid_suffix: int or None
        The blockid specified in the suffix of ``filename``. If there isn't a
        suffix, then this will be None.
    """

    # at this time, we expect the suffix to be the minimum number of characters
    # that are necessary to represent the process id. For flexibility, we will
    # allow extra zero-padding

    _dir, _base = os.path.split(filename)
    match _base.rpartition("."):
        case ("", ".", _):  # Cholla never writes a file like this
            raise ValueError(
                f"1st character in {filename!r} is the only '.' in the file's name"
            )
        case (prefix, ".", suffix) if suffix.isdecimal():
            return os.path.join(_dir, f"{prefix}.{{blockid}}"), int(suffix)
        case _:
            return (filename, None)


def _detect_particle_fields(
    dset_mapping: _DatasetDiskMapping,
) -> list[tuple[ParticleType, str]]:
    # we insert a few assert statements to flag areas of code that need to be changed
    # if/when we add support for more particle-types

    if len(dset_mapping.particle_types) == 0:
        return []
    assert dset_mapping.particle_types == ("io",)
    ptype = dset_mapping.particle_types[0]

    path = dset_mapping.particle_mapping.fname_template.format(blockid=0)
    with h5py.File(path, mode="r") as h5f:
        grp = h5f[dset_mapping.particle_mapping.h5_group_map[ptype]]
        if "n_particles_local" in grp.attrs:
            local_pfield_shape = (grp.attrs["n_particles_local"][0],)
        else:
            local_pfield_shape = (grp.attrs["total_ptype_count"][0],)

        out = []
        for name, dataset in grp.items():
            assert isinstance(dataset, h5py.Dataset)
            if dataset.shape != local_pfield_shape:
                # ensures that we don't include "density," which is a 3D field
                # holding density deposition
                continue
            out.append((ptype, name))
        return out

