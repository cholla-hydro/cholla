"""
This module is defines machinery for generating synthetic HDF5 datasets to
use for testing purposes.

Why this exists
---------------
This package is intended to be compatible with many different Cholla file
formats. But, at the time of writing, we don't have a great way to get a
bunch of datasets for testing. The best approaches to get real datasets
would be to either
- checkout, build, and create datasets using different revisions of Cholla.
  But, that approach is both
  1. not a portable solution (it can only be done on systems with GPUs)
  2. would take a long time to run the tests.
- cache example dataset outputs and download them when we need them

While the second approach is very doable, it's important to keep in mind that
we should be doing the same kind of testing for the yt-frontend.
"""

import dataclasses
import enum
import functools
import os
from collections import defaultdict
from collections.abc import Mapping, Iterable, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import h5py

from ._misc import ChollaDataFmt, _CachedH5Openner, ParticleType


@dataclasses.dataclass(frozen=True, kw_only=True)
class DomainInfo:
    """Characterizes the domain of the synthetic simulation"""

    domain_left_edge_kpc: tuple[float, float, float]
    domain_width_kpc: tuple[float, float, float]

    def __post_init__(self):
        if len(self.domain_left_edge_kpc) != 3:
            raise ValueError("domain_left_edge_kpc must be a 3 element tuple")
        elif len(self.domain_width_kpc) != 3:
            raise ValueError("domain_width_kpc must be a 3 element tuple")
        elif any(e <= 0 for e in self.domain_width_kpc):
            raise ValueError("domain_width_kpc must contain positive values")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimTimingInfo:
    """Specifies the properties pertaining to simulation time."""

    dt: float
    dt_particles: float
    t: float
    t_particles: float
    n_step: int


def get_sample_timing_info(i) -> SimTimingInfo:
    # these were all based on real-world examples
    examples = (
        SimTimingInfo(dt=0.0, dt_particles=0.0, t=0.0, t_particles=0.0, n_step=0),
        SimTimingInfo(
            dt=0.0314923, dt_particles=0.624046, t=100.0, t_particles=100.0, n_step=203
        ),
        SimTimingInfo(
            dt=0.306745, dt_particles=0.426173, t=100.0, t_particles=100.0, n_step=195
        ),
    )
    return examples[i]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SyntheticDatasetInfo:
    """Tracks properties of a synthetic Cholla dataset"""

    # Specifies the number of processes that the hypothetical Cholla simulation used
    nprocs: tuple[int, int, int]
    # Specifies the global shape of each field in the hypothetical Cholla simulation
    # (each MPI process would be responsible for evolving a subsection of the global
    # shape).
    global_shape: tuple[int, int, int]
    # Specifies domain information about the synthetic simulation
    domain_info: DomainInfo
    # Specifies information about the simulation time
    timing_info: SimTimingInfo

    def __post_init__(self):
        # a lot of these checks were inherited from when these attributes
        # were direct function arguments

        if len(self.nprocs) != 3:
            raise ValueError("nprocs must be a 3 element array")
        elif any(int(e) != e for e in self.nprocs):
            raise ValueError("nprocs must contain integers")
        elif any(e < 1 for e in self.nprocs):
            raise ValueError("nprocs must contain positive values")
        else:
            object.__setattr__(self, "nprocs", tuple(int(e) for e in self.nprocs))

        if len(self.global_shape) != 3:
            raise ValueError("global_shape must be a 3 element array")
        elif any(int(e) != e for e in self.global_shape):
            raise ValueError("global_shape must contain integers")
        elif any(e < 1 for e in self.global_shape):
            raise ValueError("global_shape must contain positive values")
        else:
            object.__setattr__(
                self, "global_shape", tuple(int(e) for e in self.global_shape)
            )

        # infer the shape of each block (and perform a sanity check)
        cc_block_shape, remainder = np.divmod(self.global_shape, self.nprocs)
        if (cc_block_shape == 0).any():
            raise ValueError(
                "nprocs contains a value exceeding the corrsponding length in "
                "global_shape"
            )
        elif (remainder != 0).any():
            raise ValueError(
                "at least 1 element of global_shape isn't evenly divisible by nprocs"
            )

    @functools.cached_property
    def cc_block_shape(self) -> tuple[int, int, int]:
        """The cell-centered block shape"""
        return tuple(np.floor_divide(self.global_shape, self.nprocs))

    @property
    def n_blocks(self):
        return int(np.prod(self.nprocs))

    @functools.cached_property
    def blockid_location_array(self):
        return np.arange(self.n_blocks).reshape(self.nprocs)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BlockItrInfo:
    blockid: int
    # selects the region of a gloabl array (representing a cell-centered field) that
    # corresponds to the current block
    cc_src_slice: tuple[slice, slice, slice]


def block_info_iterate(dset_info: SyntheticDatasetInfo) -> Iterable[BlockItrInfo]:
    cc_block_shape = dset_info.cc_block_shape
    for idx3d, blockid in np.ndenumerate(dset_info.blockid_location_array):
        cc_src_slc = (
            slice(idx3d[0] * cc_block_shape[0], (idx3d[0] + 1) * cc_block_shape[0]),
            slice(idx3d[1] * cc_block_shape[1], (idx3d[1] + 1) * cc_block_shape[1]),
            slice(idx3d[2] * cc_block_shape[2], (idx3d[2] + 1) * cc_block_shape[2]),
        )
        yield BlockItrInfo(blockid=blockid, cc_src_slice=cc_src_slc)


def _add_standard_header_attrs(f: h5py.File, timing_info: SimTimingInfo):
    # we could customize this quite a bit... (but, that seems unnecessary)

    # fields that must be handled separately:
    # - particle & field headers always have:
    #   * "bounds", "dx", "domain" (handled by _write_domain_prop_attrs)
    #   * "dims", "n_fields", "nprocs"
    #   * sometimes: "offset", "dims_local" (depends on the style)
    # - particle headers may also have:
    #   * "dt_particles" (this can be different from "dt")
    #   * "t_particles" (as far as I can tell, this is the same as "t")
    #   * sometimes: "n_particles_local" (depends on the file-style)

    f.attrs["Git Commit Hash"] = np.array(["<garbage>"], dtype=object)
    f.attrs["Macro Flags"] = np.array(["<garbage>"], dtype=object)
    f.attrs["cholla"] = np.array([""], dtype=object)
    f.attrs["density_unit"] = np.array([6.76810999e-32], dtype="f8")
    f.attrs["energy_unit"] = np.array([6.47112563e-10], dtype="f8")
    f.attrs["gamma"] = np.array([1.66666667], dtype="f8")
    f.attrs["length_unit"] = np.array([3.08567758e21], dtype="f8")
    f.attrs["mass_unit"] = np.array([1.98847e33], dtype="f8")
    f.attrs["time_unit"] = np.array([3.15569e10], dtype="f8")
    f.attrs["velocity_unit"] = np.array([9.77813911e10], dtype="f8")
    f.attrs["n_step"] = np.array([timing_info.n_step], dtype="i4")
    f.attrs["t"] = np.array([timing_info.t], dtype="f8")
    f.attrs["dt"] = np.array([timing_info.dt], dtype="f8")


def _write_domain_prop_attrs(
    f: h5py.File, global_shape: tuple[int, ...], domain_info: DomainInfo
):
    # reminder, Cholla's code-length is 1 kpc
    domain = np.array(domain_info.domain_width_kpc, dtype="f8")
    bounds = np.array(domain_info.domain_left_edge_kpc, dtype="f8")

    f.attrs["dx"] = domain / np.array(global_shape)
    f.attrs["domain"] = domain
    f.attrs["bounds"] = bounds


def _common_h5_setup(
    f: h5py.File,
    block_info: BlockItrInfo,
    dset_info: SyntheticDatasetInfo,
    data_format: ChollaDataFmt,
    n_fields: int,
):
    """
    Perform generic HDF5 setup (independent of the kind of data bein written)
    """
    # handle format-specific stuff
    match data_format:
        case ChollaDataFmt.DISTRIBUTED:
            f.attrs["offset"] = np.array(
                [int(slc.start) for slc in block_info.cc_src_slice]
            )
            f.attrs["dims_local"] = np.array(dset_info.cc_block_shape)
        case ChollaDataFmt.LEGACY_CONCAT:
            pass
        case ChollaDataFmt.CONCAT:
            blockid_location_arr = dset_info.blockid_location_array
            f.create_group("domain")
            f["domain"]["blockid_location_arr"] = blockid_location_arr
            f["domain"]["stored_blockid_list"] = np.arange(blockid_location_arr.size)
        case _:
            raise RuntimeError(f"unknown data format: {data_format}")

    # write some common metadata
    f.attrs["dims"] = np.array(dset_info.global_shape)
    f.attrs["nprocs"] = np.array(dset_info.nprocs)
    f.attrs["n_fields"] = np.array([n_fields])
    _add_standard_header_attrs(f, timing_info=dset_info.timing_info)
    _write_domain_prop_attrs(
        f,
        global_shape=dset_info.global_shape,
        domain_info=dset_info.domain_info,
    )


def _check_required_particle_props(ptype_name, ptype_props):
    _pairs = [
        ("particle_IDs", False),
        ("pos_x", True),
        ("pos_y", True),
        ("pos_z", True),
    ]

    for prop_name, expect_float in _pairs:
        try:
            dtype = ptype_props[prop_name].dtype
        except KeyError:
            raise ValueError(
                f"'{ptype_name}' particle type must have the '{prop_name}' property"
            ) from None

        if expect_float:
            parent_dtype, descr = np.floating, "a floating point value"
        else:
            parent_dtype, descr = np.integer, "an integer value"

        if not np.issubdtype(dtype, parent_dtype):
            raise ValueError(
                f"'{prop_name}' property for '{ptype_name}' particles must be {descr}"
            )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ParticleData:
    """
    Encapsulates global particle data of all particle types for a synthetic
    dataset.
    """

    # the keys of the outter mapping are particle types and the inner mapping
    # maps attributes to arrays
    data: MappingProxyType[str, MappingProxyType[str, np.ndarray]]

    def __post_init__(self):
        _required_propname_expectfloat_pairs = [
            ("particle_IDs", False),
            ("pos_x", True),
            ("pos_y", True),
            ("pos_z", True),
        ]

        for ptype, ptype_props in self.data.items():
            _check_required_particle_props(ptype, ptype_props)

            n_particles = ptype_props["particle_IDs"].size
            for prop_name, prop_vals in ptype_props.items():
                if prop_vals.size != n_particles:
                    raise ValueError(
                        f"'particle_IDs' and '{prop_name}' have unequal numbers of "
                        f"entries for the '{ptype}' particle type"
                    )

    def get_prop_name_dtype_pairs(self, ptype: str) -> list[tuple[str, np.dtype]]:
        # get (name, dtype) pairs for each property of a particle with a give type
        return [(k, arr.dtype) for k, arr in self.data[ptype].items()]

    def get_ptype_prop_pairs(self) -> list[tuple[str, str]]:
        # purely for testing api functions
        out = []
        for ptype, prop_map in self.data.items():
            out.extend((ptype, prop) for prop in prop_map)
        return out

    @property
    def n_ptypes(self):
        # number of particle types
        return len(self.data)

    def n_particles(self, ptype: str) -> int:
        # number of particles with a given particle type
        return int(np.size(self.data[ptype]["particle_IDs"]))


def identify_particle_blockids(
    particle_data: ParticleData,
    dset_info: SyntheticDatasetInfo,
) -> dict[str, np.ndarray]:
    """
    Identify the blocks containing each particle (or the block closest to the particle
    if the particle lives outside of the simulation domain).
    """
    out = {}
    for ptype in particle_data.data:
        index_components = []
        for i, axis in enumerate("xyz"):
            pos_component = particle_data.data[ptype][f"pos_{axis}"]

            left_edge = dset_info.domain_info.domain_left_edge_kpc[i]
            width = dset_info.domain_info.domain_width_kpc[i]
            blocks_per_ax = dset_info.nprocs[i]
            if blocks_per_ax == 1:
                index_components.append(np.zeros(shape=pos_component.shape, dtype="i8"))
                continue

            block_edges = (
                left_edge + np.arange(blocks_per_ax + 1) * width / blocks_per_ax
            )
            block_edges[-1] = left_edge + width  # <- just to be safe

            # at this point, block_edges has a shape of (blocks_per_ax + 1,)
            edges_between_blocks = block_edges[1:-1]
            index_components.append(
                # -> index 0 means that a particle lies to the left of
                #    edges_between_blocks[0]. In other words, it lies within the
                #    leftmost block or to the left of the leftmost block
                # -> index (blocks_per_ax-1) means that a particle lies to the right of
                #    edges_between_blocks[-1]. In other words, it lies within the
                #    rightmost block or to the right of the rightmost block
                np.digitize(pos_component, bins=edges_between_blocks, right=True)
            )
        out[ptype] = dset_info.blockid_location_array[tuple(index_components)]
    return out


class MeshDataWriter:
    @staticmethod
    def _h5group_destslc(
        block_info: BlockItrInfo, data_format: ChollaDataFmt
    ) -> tuple[str, tuple]:
        match data_format:
            case ChollaDataFmt.DISTRIBUTED:
                hdf5_grp, dst_idx = "/", (...,)
            case ChollaDataFmt.LEGACY_CONCAT:
                hdf5_grp, dst_idx = "/", (...,)
            case ChollaDataFmt.CONCAT:
                hdf5_grp, dst_idx = "field", (block_info.blockid, ...)
            case _:
                raise RuntimeError(f"unknown data format: {data_format}")
        return hdf5_grp, dst_idx

    @staticmethod
    def setup(
        f: h5py.File,
        block_info: BlockItrInfo,
        dset_info: SyntheticDatasetInfo,
        data_format: ChollaDataFmt,
        full_domain_data: dict[str, np.ndarray],
    ):
        """
        Perform setup pertaining to field files after HDF5 file is created.
        """
        field_grp, _ = MeshDataWriter._h5group_destslc(block_info, data_format)
        if field_grp != "/":
            f.create_group(field_grp)

        match data_format:
            case ChollaDataFmt.DISTRIBUTED:
                dset_shape = dset_info.cc_block_shape
            case ChollaDataFmt.LEGACY_CONCAT:
                dset_shape = dset_info.cc_block_shape
            case ChollaDataFmt.CONCAT:
                dset_shape = (np.prod(dset_info.nprocs),) + dset_info.cc_block_shape
            case _:
                raise RuntimeError(f"unknown data format: {data_format}")
        for field_name, data in full_domain_data.items():
            f[field_grp].create_dataset(
                name=field_name, shape=dset_shape, dtype=data.dtype
            )

    @staticmethod
    def write(
        f: h5py.File,
        block_info: BlockItrInfo,
        data_format: ChollaDataFmt,
        full_domain_data: dict[str, np.ndarray],
        general_selection_ctx: None,
    ):
        """
        Actually write the relevant data to the provided file
        """
        assert general_selection_ctx is None
        field_grp, dst_idx = MeshDataWriter._h5group_destslc(block_info, data_format)
        src_slc = block_info.cc_src_slice
        for field_name in full_domain_data:
            f[field_grp][field_name][dst_idx] = full_domain_data[field_name][src_slc]


class ParticleDataWriter:
    @staticmethod
    def _create_h5_datasets(
        h5_grp: h5py.Group,
        property_name_dtype_pairs: Iterable[tuple[str, np.dtype]],
        particle_count: int,
    ):
        for prop_name, dtype in property_name_dtype_pairs:
            h5_grp.create_dataset(name=prop_name, dtype=dtype, shape=(particle_count,))

    @staticmethod
    def setup(
        f: h5py.File,
        block_info: BlockItrInfo,
        dset_info: SyntheticDatasetInfo,
        data_format: ChollaDataFmt,
        full_domain_data: ParticleData,
    ):
        timing_info = dset_info.timing_info
        f.attrs["t_particles"] = np.array([timing_info.t_particles], dtype="f8")
        f.attrs["dt_particles"] = np.array([timing_info.dt_particles], dtype="f8")
        match data_format:
            case ChollaDataFmt.DISTRIBUTED:
                # we'll create the datasets later (when we know exactly how many
                # particles to write out per file)
                pass

            case ChollaDataFmt.LEGACY_CONCAT:
                raise ValueError("no legacy concat support for particle data")

            case ChollaDataFmt.CONCAT:
                grp = f.create_group("particle")
                n_blocks = dset_info.n_blocks

                for ptype_name in full_domain_data.data:
                    ptype_grp = grp.create_group(ptype_name)
                    ptype_count = full_domain_data.n_particles(ptype=ptype_name)

                    # create standard attribute and dataset
                    ptype_grp.attrs["total_ptype_count"] = ptype_count
                    tmp_h5dset = ptype_grp.create_dataset(
                        name="stop_block_idx_slc", shape=(n_blocks,), dtype="i8"
                    )
                    tmp_h5dset[...] = 0

                    # create datasets that may very for different particle types
                    ParticleDataWriter._create_h5_datasets(
                        ptype_grp,
                        full_domain_data.get_prop_name_dtype_pairs(ptype_name),
                        ptype_count,
                    )
            case _:
                raise RuntimeError(f"unknown data format: {data_format}")

    @staticmethod
    def write(
        f: h5py.File,
        block_info: BlockItrInfo,
        data_format: ChollaDataFmt,
        full_domain_data: ParticleData,
        general_selection_ctx: dict[str, np.ndarray],
    ):
        block_id = block_info.blockid
        for ptype_name in full_domain_data.data:
            local_particle_mask = general_selection_ctx[ptype_name] == block_id
            local_particle_count = np.sum(local_particle_mask)

            match data_format:
                case ChollaDataFmt.DISTRIBUTED:
                    if full_domain_data.n_ptypes > 1:
                        raise ValueError(
                            "The distributed format isn't defined when there are "
                            "multiple particle types"
                        )
                    f.attrs["n_particles_local"] = np.array(
                        [local_particle_count], dtype="i8"
                    )
                    ParticleDataWriter._create_h5_datasets(
                        f,
                        full_domain_data.get_prop_name_dtype_pairs(ptype_name),
                        particle_count=local_particle_count,
                    )

                    grp = f
                    dst_slice = (...,)

                case ChollaDataFmt.LEGACY_CONCAT:
                    raise ValueError("no legacy concat support for particle data")

                case ChollaDataFmt.CONCAT:
                    # we already created the datasets in this case
                    grp = f["particle"][ptype_name]
                    if block_id == 0:
                        dst_start = 0
                    else:
                        dst_start = grp["stop_block_idx_slc"][block_id - 1]
                    dst_slice = slice(dst_start, dst_start + local_particle_count)
                    grp["stop_block_idx_slc"][block_id] = dst_slice.stop

            if local_particle_count == 0:
                continue

            for prop_name, arr in full_domain_data.data[ptype_name].items():
                grp[prop_name][dst_slice] = arr[local_particle_mask]


def _create_fname_template(
    root_path: str, data_format: ChollaDataFmt, particle: bool
) -> str:
    """
    Create the filename template.

    Parameters
    ----------
    root_path
        Prefix of the path where the dataset is written
    data_format
        Specifies the format of the dataset
    particle
        Specifies whether the file primarily is for particle data

    Returns
    -------
    fname_template: str
        Calling `fname_template.format(block_id=block_id)` produces the path
        to the file where data associated with block_id gets saved.
    """
    root_particle_suffix = "_particles"

    if len(root_path) == 0 or root_path[-1] in (os.sep, os.altsep):
        raise ValueError(
            "root path can't simply be an empty string or the name of a directory "
            "followed by path-component separator. It must include at least 1 "
            "character to act as the prefix for the start of the filename."
        )
    elif root_path.endswith(root_particle_suffix):
        raise ValueError(f"root_path should not end with `{root_particle_suffix}`")
    elif particle:
        root_path = root_path + root_particle_suffix

    match data_format:
        case ChollaDataFmt.DISTRIBUTED:
            return f"{root_path}.h5.{{blockid:d}}"
        case ChollaDataFmt.LEGACY_CONCAT:
            return f"{root_path}.h5"
        case ChollaDataFmt.CONCAT:
            return f"{root_path}.h5"
        case _:
            raise RuntimeError(f"unknown data format: {data_format}")


def generate_files(
    root_path: str,
    dset_info: SyntheticDatasetInfo,
    data_format: ChollaDataFmt | tuple[ChollaDataFmt, ChollaDataFmt],
    field_data: dict[str, np.ndarray] | None,
    particle_data: ParticleData | None = None,
    write_single_file: bool = False,
) -> tuple[str, str]:
    """
    Generates file(s) that emulate a dataset holding results from a
    hypothetical Cholla simulation

    Parameters
    ----------
    root_path
        Prefix of the path where the dataset is written
    dset_info
        Specifies properties number of the synthetic dataset
    data_format
        Specifies the format of the dataset
    field_data
        An optional dictionary mapping field names to arrays that represent
        the global concatenated fields
    particle_data
        Optionally specifies a ParticleData instance that holds globally
        concatenated particle data

    Returns
    -------
    field_root_fname, particle_root_fname: str or None
        Path to one of the generated files holding field or particle data. For
        distributed datasets this always corresponds to block 0.
    """
    # check args and coerce data_format to a 2-tuple
    match (field_data, particle_data, data_format):
        case (None, None, _, _):
            raise ValueError("field_data & particle_data are both None")
        case (None, _, ChollaDataFmt()) | (_, None, ChollaDataFmt()):
            data_format = (data_format, data_format)
        case (None, _, _) | (_, None, _):
            raise ValueError(
                "if field_data or particle_data is None, data_format must be a "
                "ChollaDataFmt"
            )
        case (_, _, ChollaDataFmt()):
            data_format = (data_format, data_format)
        case (_, _, [ChollaDataFmt(), ChollaDataFmt()]):
            data_format = tuple(data_format)
        case _:
            raise TypeError("data_format must be a ChollaDataFmt or ChollaDataFmt pair")

    disallow_wsf = ((field_data is None) or (particle_data is None)) or data_format != (
        ChollaDataFmt.CONCAT,
        ChollaDataFmt.CONCAT,
    )
    if write_single_file and disallow_wsf:
        raise ValueError(
            "for write_single_file to be true, particle_data & field_data must both "
            "(i) be provided and (ii) be associated with concat data format"
        )

    # overwrite dset_info before defining _write (since it's captured by _write)
    if data_format[0] is ChollaDataFmt.LEGACY_CONCAT:
        dset_info = dataclasses.replace(dset_info, nprocs=(1, 1, 1))

    def _write(fname_template, n_fields, data_fmt, triple_l):
        # helper function that actually writes data to disk
        with _CachedH5Openner(mode="w") as h5_manager:
            for b_info in block_info_iterate(dset_info):
                f = h5_manager.open_fh(fname_template.format(blockid=b_info.blockid))
                kw = {"f": f, "block_info": b_info, "data_format": data_fmt}

                is_new_file = (b_info.blockid == 0) or (not data_fmt.is_single_file)
                if is_new_file:
                    _common_h5_setup(dset_info=dset_info, n_fields=n_fields, **kw)

                for writer, d, ctx in triple_l:
                    if is_new_file:
                        writer.setup(dset_info=dset_info, full_domain_data=d, **kw)
                    writer.write(full_domain_data=d, general_selection_ctx=ctx, **kw)
        return fname_template.format(blockid=0)

    n_fields = 0
    triples_l = [None, None]
    if field_data is not None:
        n_fields = len(field_data)
        triples_l[0] = (MeshDataWriter, field_data, None)

    if particle_data is not None:
        if data_format[1] is ChollaDataFmt.LEGACY_CONCAT:
            raise ValueError("no legacy concat support for particle data")
        particle_blockid_map = identify_particle_blockids(particle_data, dset_info)
        triples_l[1] = (ParticleDataWriter, particle_data, particle_blockid_map)

    out_fname_l = [None, None]
    if write_single_file:
        template = _create_fname_template(root_path, ChollaDataFmt.CONCAT, False)
        tmp = _write(template, n_fields, ChollaDataFmt.CONCAT, triples_l)
        out_fname_l[0] = tmp
        out_fname_l[1] = tmp

    else:
        if triples_l[0] is not None:  # writing field data
            template = _create_fname_template(root_path, data_format[0], False)
            out_fname_l[0] = _write(template, n_fields, data_format[0], triples_l[:1])
        if triples_l[1] is not None:  # writing particle data
            template = _create_fname_template(root_path, data_format[1], True)
            out_fname_l[1] = _write(template, n_fields, data_format[1], triples_l[1:])

    return out_fname_l[0], out_fname_l[1]


def _generate_array(shape: tuple[int, ...], *, start: int = 0):
    # used to generate an array of unique values of a given shape
    size = np.prod(shape)
    return np.arange(start, start + size).reshape(shape).astype("f8")


def _generate_field_data(field_names: Sequence[str], global_shape: tuple[int, ...]):
    # this generates field data (ensuring that every element of each field has a
    # totally unique value)

    # construct the global arrays
    global_arrays = {}
    start_offset = 0
    for name in field_names:
        arr = _generate_array(global_shape, start=start_offset)
        global_arrays[name] = arr
        start_offset = arr.max() + 1
    return global_arrays


def _generate_particle_data(domain_info: DomainInfo) -> ParticleData:
    # this is extremely simplistic, but it gets the job done

    rng = np.random.default_rng(12345)
    fractional_positions = rng.random((300, 3), dtype="f8")

    _global_props = defaultdict(list)
    for particle_id, fractional_pos in enumerate(fractional_positions):
        _global_props["particle_IDs"].append(particle_id)
        for i, ax in enumerate("xyz"):
            _global_props[f"pos_{ax}"].append(
                domain_info.domain_left_edge_kpc[i]
                + fractional_pos[i] * domain_info.domain_width_kpc[i]
            )

    def _dtype(name):
        if name == "particle_IDs":
            return "i8"
        return "f8"

    props = MappingProxyType(
        {k: np.array(v, dtype=_dtype(k)) for k, v in _global_props.items()}
    )

    return ParticleData(data=MappingProxyType({"io": props}))


class ExtraFmt(enum.Enum):  # supplements ChollaDataFmt
    DistribF_ConcatP = enum.auto()
    ConcatF_DistribP = enum.auto()
    UnifiedConcat = enum.auto()  # <- write fields and particles to a single file


class BlockPreset(enum.Enum):
    """
    Decribes the properties that control the properties of the blocks.

    Really, we specifying the domain shape and the partitioning, but I can't think
    of a good name for that
    """

    # name each preset
    # -> the assigned value is (enum-value, dict(nprocs=<val>, global_shape=<val>))
    CubeDomain_1 = (enum.auto(), {"nprocs": (1, 1, 1), "global_shape": (8, 8, 8)})
    IrrDomain_222 = (enum.auto(), {"nprocs": (2, 2, 2), "global_shape": (4, 16, 8)})
    IrrDomain_142 = (enum.auto(), {"nprocs": (1, 4, 2), "global_shape": (4, 16, 8)})

    def __new__(cls, val: Any, properties: dict[str, tuple[int, int, int]]):
        # based on example from docs
        if isinstance(val, enum.auto):
            val = len(cls.__members__) + 1

        obj = object.__new__(cls)
        obj._value_ = val
        obj.properties = MappingProxyType(properties)
        return obj

    def __repr__(self):  # based on example from docs (hide the underlying value)
        return f"<{self.__class__.__name__}, {self.name}>"


@functools.cache
def _make_dset_info_preset(preset: BlockPreset) -> SyntheticDatasetInfo:
    return SyntheticDatasetInfo(
        nprocs=preset.properties["nprocs"],
        global_shape=preset.properties["global_shape"],
        domain_info=DomainInfo(
            domain_left_edge_kpc=(-25.0, -25.0, -25.0),
            domain_width_kpc=(100.0, 100.0, 100.0),
        ),
        timing_info=get_sample_timing_info(1),
    )


# define a constant (its meant to be used similarly to None)


class UseDefaultType(enum.Enum):
    USE_DEFAULT = 0


USE_DEFAULT = UseDefaultType.USE_DEFAULT


@dataclasses.dataclass(frozen=True)
class SyntheticInputPack:
    """
    Groups high-level inputs for generating a synthetic snapshot

    This class is intended to be use for parameterizing inputs in pytest
    unit-tests. We have explicitly created a __repr__ method that will ensure
    that the resulting tests will be nice and readable.
    """

    blocking_preset: BlockPreset
    field: None | UseDefaultType | Mapping[str, np.ndarray]
    particle: None | UseDefaultType | ParticleData
    datafmt: ChollaDataFmt | ExtraFmt

    def __post_init__(self):
        if self.field is None and self.particle is None:
            raise ValueError()

    def get_field_data(self) -> Mapping[str, np.ndarray] | None:
        if self.field is None:
            return None
        elif self.field is USE_DEFAULT:
            global_shape = _make_dset_info_preset(self.blocking_preset).global_shape
            return _generate_field_data(
                field_names=["density", "momentum_x"], global_shape=global_shape
            )
        else:
            return self.field

    def get_particle_data(self) -> ParticleData | None:
        if self.particle is None:
            return None
        elif self.particle is USE_DEFAULT:
            dset_info = _make_dset_info_preset(self.blocking_preset)
            return _generate_particle_data(dset_info.domain_info)
        else:
            return self.particle

    def __repr__(self) -> str:
        """return a concise human-readible description"""
        comps = [self.blocking_preset.name]

        if self.field is USE_DEFAULT and self.particle is USE_DEFAULT:
            comps.append("DfltData")
        else:
            if self.field is None:
                f_descr = "NoField"
            elif self.field is USE_DEFAULT:
                f_descr = "DfltField"
            else:
                f_descr = "CustomField"
            if self.particle is None:
                p_descr = "NoParticle"
            elif self.particle is USE_DEFAULT:
                p_descr = "DfltParticle"
            else:
                p_descr = "CustomParticle"
            comps.append(f"{f_descr}_{p_descr}")

        comps.append(self.datafmt.name)
        return ":".join(comps)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SyntheticSnapResult:
    """
    Summarizes details about a synthetic snapshot.

    The basic premise is that we will use a pytest fixture to construct a
    synthetic dataset from a SyntheticInputPack instance, and an instance of
    this type will be handed to a unit test to describe the synthetic snapshot
    """

    # the input used to create this instance
    input_pack: SyntheticInputPack

    # The more detailed Inputs
    # ------------------------
    # the synthetic dataset saved to disk
    dset_info: SyntheticDatasetInfo
    # dictionary of any globally concatenated fields that were saved
    field_data: dict[str, np.ndarray] | None
    # holds any particle data that was saved
    particle_data: ParticleData | None

    # The Outputs
    # -----------
    # path to the root file holding field data (if any)
    root_field_fname: str | None
    # path to the root file holding particle data (if any)
    root_particle_fname: str | None


def create_snap_from_input_pack(
    root_path: str, input_pack: SyntheticInputPack
) -> SyntheticSnapResult:
    common_kw = {
        "dset_info": _make_dset_info_preset(input_pack.blocking_preset),
        "field_data": input_pack.get_field_data(),
        "particle_data": input_pack.get_particle_data(),
    }

    write_single_file = False
    match input_pack.datafmt:
        case ChollaDataFmt():
            data_format = input_pack.datafmt
        case ExtraFmt.DistribF_ConcatP:
            data_format = (ChollaDataFmt.DISTRIBUTED, ChollaDataFmt.CONCAT)
        case ExtraFmt.ConcatF_DistribP:
            data_format = (ChollaDataFmt.CONCAT, ChollaDataFmt.DISTRIBUTED)
        case ExtraFmt.UnifiedConcat:
            data_format = ChollaDataFmt.CONCAT
            write_single_file = True
        case _:
            raise RuntimeError("should be unreachable")

    root_field_fname, root_particle_fname = generate_files(
        root_path=root_path,
        data_format=data_format,
        write_single_file=write_single_file,
        **common_kw,
    )
    return SyntheticSnapResult(
        input_pack=input_pack,
        root_field_fname=root_field_fname,
        root_particle_fname=root_particle_fname,
        **common_kw,
    )


def _fetch_keys(actual, reference, err_msg=""):
    # check consistency in dictionary keys
    __tracebackhide__ = True  # Hide traceback for py.test

    refkeys = reference.keys()
    refkey_set = set(refkeys)
    mismatch_keys = refkey_set.symmetric_difference(actual.keys())

    if len(mismatch_keys):
        shared_keys = list(refkey_set.intersection(actual.keys()))
        extra_ref, extra_actual = [], []
        for k in mismatch_keys:
            if k in refkeys:
                extra_ref.append(k)
            else:
                extra_actual.append(k)

        raise AssertionError(
            "The keys are not equal.\n"
            f"{err_msg}\n"
            "There is a keys mismatch. Both dicts of arrays have the keys:\n"
            f" {shared_keys!r}\n"
            "Extra Keys:\n"
            f" actual:    {extra_actual}\n"
            f" reference: {extra_ref}"
        )
    return list(refkeys)


def assert_arraydict_equal(
    actual: Mapping[str, np.ndarray], desired, err_msg="", *, strict=True
):
    """
    Raises an AssertionError if any contents of the 2 compared mappings of
    arrays are not EXACTLY equal

    Parameters
    ----------
    actual
        A mapping of actual arrays to check
    desired
        A mapping of desired arrays to check
    err_msg
        Custom error message to be printed in case of failure.
    strict
        When True, an AssertionError gets raised if the dtypes or shapes of
        compared arrays don't exactly match
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    keys = _fetch_keys(actual, desired, err_msg=err_msg)
    for key in keys:
        np.testing.assert_array_equal(
            actual[key], desired[key], err_msg=err_msg, strict=True
        )


def assert_equal_particle_data(
    actual: dict[tuple[ParticleType, str], np.ndarray],
    desired: dict[tuple[ParticleType, str], np.ndarray] | ParticleData,
    exhaustive: bool,
    err_msg: str = "",
    *,
    strict: bool = True,
):
    """
    Raises an AssertionError if two objects are not equal.

    Parameters
    ----------
    actual
        The object to check
    desired
        The expected object
    exhaustive
        When False, it is acceptable for desired to have more keys than
        actual. When True, an AssertionError is raised if the number of keys
        don't exactly match
    err_msg
        Custom error message to be printed in case of failure.
    strict
        When True, an AssertionError gets raised if the dtypes or shapes of
        compared arrays don't exactly match
    """
    # __tracebackhide__ = True  # Hide traceback for py.test

    if isinstance(desired, ParticleData):
        _data = desired.data
        desired = {}
        for ptype, prop_map in _data.items():
            for prop_name, arr in prop_map.items():
                desired[(ptype, prop_name)] = arr

    ptypes = set(ptype for (ptype, _) in actual)

    idx_pairs = {}
    for ptype in ptypes:
        try:
            idx_actual = np.argsort(actual[ptype, "particle_IDs"])
            idx_desired = np.argsort(desired[ptype, "particle_IDs"])
        except KeyError:
            raise ValueError(
                f'this function requires ("{ptype}", "particle_IDs") in both dicts'
            ) from None
        idx_pairs[ptype] = (idx_actual, idx_desired)

    if exhaustive:
        keys = _fetch_keys(actual, desired)
    else:
        keys = list(actual.keys())

    for key in keys:
        ptype, _ = key
        idx_actual, idx_desired = idx_pairs[ptype]
        a_arr = actual[key]
        try:
            d_arr = desired[key]
        except KeyError:
            raise AssertionError(
                f"the {key!r} is in actual but not desired\n{err_msg}"
            ) from None

        np.testing.assert_array_equal(
            a_arr[idx_actual], d_arr[idx_desired], err_msg=err_msg, strict=strict
        )
