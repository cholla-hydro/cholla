import subprocess

import numpy as np
import h5py
import pytest

import cholla_utils

def _generate_array(shape):
    size = np.prod(shape)
    return np.arange(size).reshape(shape).astype("f8")


def _generate_files(root_path, nprocs, field_name = "density"):
    global_shape = (2,4,8)
    global_array = _generate_array(global_shape)
    cc_block_shape = np.floor_divide(global_shape, nprocs)

    #dims_global = [1.0, 1.0, 1.0]
    #dims_local = np.divide(dims_global, nprocs)

    blockid_location_arr = np.arange(np.prod(nprocs)).reshape(nprocs)

    fnames = []

    for idx3d, blockid in np.ndenumerate(blockid_location_arr):
        layout_slc=(
            slice(idx3d[0]*cc_block_shape[0], (idx3d[0]+1)*cc_block_shape[0]),
            slice(idx3d[1]*cc_block_shape[1], (idx3d[1]+1)*cc_block_shape[1]),
            slice(idx3d[2]*cc_block_shape[2], (idx3d[2]+1)*cc_block_shape[2])
        )
        cur_fname = f"{root_path}.h5.{blockid}"
        if blockid == 0:
            root_fname = cur_fname
        with h5py.File(cur_fname, "w") as f:
            f.attrs["offset"] = np.array([int(slc.start) for slc in layout_slc])
            #f.attrs["dims"] = np.array(dims_global)
            #f.attrs["dims_local"] = np.array(dims_local)
            f.attrs["dims"] = np.array(global_array.shape)
            f.attrs["dims_local"] = cc_block_shape
            f[field_name] = global_array[layout_slc]

    return global_array, root_fname


def test_simple_scenario(tmp_path):
    field_name = "my-field"

    # TODO: try testing with distributed datasets (i.e. when nprocs is not [1, 1, 1]
    global_arr, root_fname = _generate_files(
        str(tmp_path/ "dummy"), nprocs = [1, 1, 1], field_name=field_name
    )

    loaded = cholla_utils.load_field(root_fname, field=field_name)
    np.testing.assert_equal(global_arr, loaded)


def test_simple_failure(tmp_path):
    field_name = "my-field"

    # TODO: try testing with distributed datasets (i.e. when nprocs is not [1, 1, 1]
    global_arr, root_fname = _generate_files(
        str(tmp_path/ "dummy"), nprocs = [1, 1, 1], field_name=field_name
    )
   
    with pytest.raises(KeyError) as excinfo:
        cholla_utils.load_field(root_fname, field="not-a-field")

    with pytest.raises(FileNotFoundError) as excinfo:
        cholla_utils.load_field("/not/a/file", field="not-a-field")
