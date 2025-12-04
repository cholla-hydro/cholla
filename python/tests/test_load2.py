"""
Tests the Cholla-frontend by generating synthetic datasets

In more detail, Cholla has had a couple of historical data formats.
- see the ``ChollaDataFmt`` enumeration for a description of each format
- we choose to use synthetic datasets in case there is a future where the
  data-format changes again and we want to maintain backwards compatability
  without uploading more and more sample Cholla datasets
- the logic for creating synthetic datasets is adapted from similar logic
  used for testing the ``cholla_utils`` python package
  - the ``cholla_utils`` python package is developed within the Cholla
    repository that provides a light-weight (compared to yt) interface for
    loading datasets
  - ideally, we will try to keep the testing logic relatively consistent
    between the 2 packages
"""

import dataclasses
import typing
from collections import defaultdict

import numpy as np
import pytest
import h5py

from cholla_utils._misc import ChollaDataFmt, _CachedH5Openner
import cholla_utils

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
    # the fields in the synthetic dataset. Aside: particle datafiles must record the
    # number of fields
    field_names: list[str]

    def __post_init__(self):
        # a lot of these checks were inherited from when these attributes
        # were direct function arguments

        if any(not isinstance(name, str) for name in self.field_names):
            raise TypeError("field_names must be a sequence of strings")
        elif isinstance(self.field_names, str):
            raise TypeError("field_names can't be a string")
        elif len(self.field_names) != len(set(self.field_names)):
            raise ValueError("field_names must hold unique names")

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

    @property
    def cc_block_shape(self) -> tuple[int, int, int]:
        """The cell-centered block shape"""
        return tuple(np.floor_divide(self.global_shape, self.nprocs))

    def get_blockid_location_array(self) -> np.ndarray:
        return np.arange(np.prod(self.nprocs)).reshape(self.nprocs)


def _generate_array(shape: tuple[int, ...], *, start: int = 0):
    # used to generate an array of unique values of a given shape
    size = np.prod(shape)
    return np.arange(start, start + size).reshape(shape).astype("f8")


def _add_standard_header_attrs(f: h5py.File):
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
    f.attrs["n_step"] = np.array([0], dtype="i4")
    f.attrs["t"] = np.array([0.0], dtype="f8")
    f.attrs["dt"] = np.array([0.0], dtype="f8")


def _write_domain_prop_attrs(
    f: h5py.File, global_shape: tuple[int, ...], domain_info: DomainInfo
):
    # reminder, Cholla's code-length is 1 kpc
    domain = np.array(domain_info.domain_width_kpc, dtype="f8")
    bounds = np.array(domain_info.domain_left_edge_kpc, dtype="f8")

    f.attrs["dx"] = domain / np.array(global_shape)
    f.attrs["domain"] = domain
    f.attrs["bounds"] = bounds


def _generate_global_particle_props(domain_info: DomainInfo):
    # this is extremely simplistic, but it gets the job done

    fractional_positions = [(0.35, 0.35, 0.35), (0.65, 0.35, 0.65)]

    _global_props = defaultdict(list)
    for particle_id, fractional_pos in enumerate(fractional_position):
        _global_props["particle_IDs"].append(particle_id)
        for i, ax in enumerate("xyz"):
            _global_props[f"pos_{ax}"].append(
                domain_info.domain_left_edge_kpc[i]
                + fractional_pos[i] * domain_info.domain_width_kpc[i]
            )
    return {k: np.array(v) for k, v in _global_props.items()}


@dataclasses.dataclass(frozen=True, kw_only=True)
class BlockItrInfo:
    blockid: int
    cc_src_slice: tuple[slice, slice, slice]


class MeshDataWriter:
    @staticmethod
    def creation_callback(
        f: h5py.File,
        block_info: BlockItrInfo,
        dset_info: SyntheticDatasetInfo,
        data_format: ChollaDataFmt,
    ):
        # called immediately after a file was created
        if data_format is ChollaDataFmt.CONCAT:
            f.create_group("field")
        # maybe return the output shape of each dataset?

    @staticmethod
    def write_data(
        f: h5py.File,
        block_info: BlockItrInfo,
        dset_info: SyntheticDatasetInfo,
        data_format: ChollaDataFmt,
    ):
        # called immediately after a file was created
        if data_format is ChollaDataFmt.CONCAT:
            f.create_group("field")


def _generate_files(
    root_path: str,
    dset_info: SyntheticDatasetInfo,
    data_format: ChollaDataFmt,
) -> tuple[dict[str, np.ndarray], str]:
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

    Returns
    -------
    global_arrays: dict[str, np.ndarray]
        A dictionary of the global concatenated fields
    root_fname: str
        Path to one of the files. For distributed datasets this is
        always process 0.
    """
    if data_format is ChollaDataFmt.LEGACY_CONCAT:
        dset_info = dataclasses.replace(dset_info, nprocs=(1, 1, 1))

    cc_block_shape = dset_info.cc_block_shape

    # construct the global arrays
    global_arrays = {}
    start_offset = 0
    for name in dset_info.field_names:
        arr = _generate_array(dset_info.global_shape, start=start_offset)
        global_arrays[name] = arr
        start_offset = arr.max() + 1

    # prepare to creating the files
    blockid_location_arr = dset_info.get_blockid_location_array()
    match data_format:
        case ChollaDataFmt.DISTRIBUTED:
            fname_template = f"{root_path}.h5.{{blockid:d}}"
            field_grp = "/"
            field_dset_shape = cc_block_shape
        case ChollaDataFmt.LEGACY_CONCAT:
            fname_template = f"{root_path}.h5"
            field_grp = "/"
            field_dset_shape = cc_block_shape
        case ChollaDataFmt.CONCAT:
            fname_template = f"{root_path}.h5"
            field_grp = "field"
            field_dset_shape = (np.prod(dset_info.nprocs),) + cc_block_shape
        case _:
            raise RuntimeError(f"unknown data format: {data_format}")

    # actually create the file
    with _CachedH5Openner(mode="w") as h5_context_manager:
        for idx3d, blockid in np.ndenumerate(blockid_location_arr):
            f = h5_context_manager.open_fh(fname_template.format(blockid=blockid))

            # selects the region of a global arrays relevant for the current block
            src_slc = (
                slice(idx3d[0] * cc_block_shape[0], (idx3d[0] + 1) * cc_block_shape[0]),
                slice(idx3d[1] * cc_block_shape[1], (idx3d[1] + 1) * cc_block_shape[1]),
                slice(idx3d[2] * cc_block_shape[2], (idx3d[2] + 1) * cc_block_shape[2]),
            )

            # determine the region of the output dataset relevant for the current block
            # and write any extra output-specific metadata
            match data_format:
                case ChollaDataFmt.DISTRIBUTED:
                    dst_idx = (...,)

                    f.attrs["offset"] = np.array([int(slc.start) for slc in src_slc])
                    f.attrs["dims_local"] = np.array(cc_block_shape)

                case ChollaDataFmt.LEGACY_CONCAT:
                    dst_idx = (...,)

                case ChollaDataFmt.CONCAT:
                    dst_idx = (blockid, ...)

                    if blockid == 0:
                        f.create_group("domain")
                        f["domain"]["blockid_location_arr"] = blockid_location_arr
                        f["domain"]["stored_blockid_list"] = np.arange(
                            blockid_location_arr.size
                        )

                        f.create_group("field")

                case _:
                    raise RuntimeError(f"unknown data format: {data_format}")

            if (blockid == 0) or (not data_format.is_single_file):
                # write some common metadata
                f.attrs["dims"] = np.array(dset_info.global_shape)
                f.attrs["nprocs"] = np.array(dset_info.nprocs)
                f.attrs["n_fields"] = np.array([len(dset_info.field_names)])
                _add_standard_header_attrs(f)
                _write_domain_prop_attrs(
                    f,
                    global_shape=dset_info.global_shape,
                    domain_info=dset_info.domain_info,
                )
                # create the datasets that will hold the fields
                for field_name in dset_info.field_names:
                    f[field_grp].create_dataset(
                        name=field_name, shape=field_dset_shape, dtype="f8"
                    )

            # actually record the field data
            for field_name in dset_info.field_names:
                f[field_grp][field_name][dst_idx] = global_arrays[field_name][src_slc]
    return global_arrays, fname_template.format(blockid=0)


def _get_common_dset_info_objs():
    kwargs = {
        "domain_info": DomainInfo(
            domain_left_edge_kpc=(-25.0, -25.0, -25.0),
            domain_width_kpc=(100.0, 100.0, 100.0),
        ),
        "field_names": ["density", "momentum_x"],
    }

    return (
        SyntheticDatasetInfo(nprocs=(1, 1, 1), global_shape=(8, 8, 8), **kwargs),
        SyntheticDatasetInfo(nprocs=(2, 2, 2), global_shape=(4, 16, 8), **kwargs),
        SyntheticDatasetInfo(nprocs=(1, 4, 2), global_shape=(4, 16, 8), **kwargs),
    )


_COMMON_DSET_INFO_OBJS = _get_common_dset_info_objs()


_CASES = [
    {
        "dset_info": _COMMON_DSET_INFO_OBJS[0],
        "data_format": ChollaDataFmt.DISTRIBUTED,
    },
    {
        "dset_info": _COMMON_DSET_INFO_OBJS[1],
        "data_format": ChollaDataFmt.DISTRIBUTED,
    },
    {
        "dset_info": _COMMON_DSET_INFO_OBJS[2],
        "data_format": ChollaDataFmt.DISTRIBUTED,
    },
    # there no point going through lots of varieties of ChollaDataFmt.LEGACY_CONCAT
    # -> the files always look very similar to each other
    {
        "dset_info": _COMMON_DSET_INFO_OBJS[0],
        "data_format": ChollaDataFmt.LEGACY_CONCAT,
    },
    # it's definitely worth checking ChollaDataFmt.LEGACY_CONCAT when there is only
    # 1 process as well as when there are multiple processes
    {
        "dset_info": _COMMON_DSET_INFO_OBJS[0],
        "data_format": ChollaDataFmt.CONCAT,
    },
    {
        "dset_info": _COMMON_DSET_INFO_OBJS[1],
        "data_format": ChollaDataFmt.CONCAT,
    },
    {
        "dset_info": _COMMON_DSET_INFO_OBJS[2],
        "data_format": ChollaDataFmt.CONCAT,
    },
]


@pytest.mark.parametrize("kwargs", _CASES)
def test_load(tmp_path, kwargs):
    # generate a synthetic dataset and make sure that the loaded values are correct

    # Step 1: create the synthetic dataset
    # -> global_arr is a dict that maps each field_name to an array that holding the
    #    fully concatenated array that holds the expected field values
    # -> root_fname is the path that should be passed to yt.load
    global_arr, root_fname = _generate_files(root_path=str(tmp_path / "0"), **kwargs)
    field_names = list(global_arr.keys())

    # Step 2: try loading a single field
    loaded = cholla_utils.load_field(root_fname, field=field_names[-1])
    np.testing.assert_equal(global_arr[field_names[-1]], loaded)

    # Step 3: try loading all field names
    loaded = cholla_utils.load_field(root_fname, field=field_names)
    for name in field_names:
        np.testing.assert_equal(global_arr[name], loaded[name])

