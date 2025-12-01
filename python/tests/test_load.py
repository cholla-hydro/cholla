import os
import subprocess
import sys
from collections.abc import Sequence
from enum import Enum, auto
from tempfile import TemporaryDirectory

import numpy as np
import h5py
import pytest

import cholla_utils

_PYTHON_SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts")


# declare a global constant used to represent the legacy concatenation format
class ConcatFormat(Enum):
    Legacy = auto()
    Modern = auto()


def _generate_array(shape, *, start=0):
    size = np.prod(shape)
    return np.arange(start, start + size).reshape(shape).astype("f8")


def _generate_files(
    root_path: str,
    nprocs: Sequence[int],
    global_shape: Sequence[int],
    *,
    concat_format: ConcatFormat | None = None,
    field_names: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    """This function generates 1 or more mock-files (for testing purposes)

    Returns
    -------
    global_arrays: dict[str, np.ndarray]
        A dictionary of the global concatenated fields
    root_fname: str
        Path to one of the files. For distributed datasets this is
        always process 0.
    """
    # check and sanitize arguments

    if isinstance(field_names, str):
        raise TypeError("field_names can't be a string")
    elif field_names is None:
        field_names = ["density"]
    elif len(field_names) != len(set(field_names)):
        raise ValueError("field_names must hold unique names")

    if len(nprocs) != 3:
        raise ValueError("nprocs must be a 3 element array")
    elif any(int(e) != e for e in nprocs):
        raise ValueError("nprocs must contain integers")
    elif any(e < 1 for e in nprocs):
        raise ValueError("nprocs must contain positive values")
    else:
        nprocs = tuple(int(e) for e in nprocs)

    if len(global_shape) != 3:
        raise ValueError("global_shape must be a 3 element array")
    elif any(int(e) != e for e in global_shape):
        raise ValueError("global_shape must contain integers")
    elif any(e < 1 for e in global_shape):
        raise ValueError("global_shape must contain positive values")
    else:
        global_shape = tuple(int(e) for e in global_shape)

    # infer the shape of each block (and perform a sanity check)
    cc_block_shape, remainder = np.divmod(global_shape, nprocs)
    if (cc_block_shape == 0).any():
        raise ValueError(
            "nprocs contains a value exceeding the corrsponding length in global_shape"
        )
    elif (remainder != 0).any():
        raise ValueError(
            "at least 1 element of global_shape isn't evenly divisible by nprocs"
        )

    if concat_format is ConcatFormat.Legacy:
        cc_block_shape = np.array(global_shape)
        nprocs = (1, 1, 1)

    # construct the global arrays
    global_arrays = {}
    start_offset = 0
    for name in field_names:
        arr = _generate_array(global_shape, start=start_offset)
        global_arrays[name] = arr
        start_offset = arr.max() + 1

    # define the functionality for creating the files
    def _work(fname_prefix):
        blockid_location_arr = np.arange(np.prod(nprocs)).reshape(nprocs)
        root_fname = None
        for idx3d, blockid in np.ndenumerate(blockid_location_arr):
            layout_slc = (
                slice(idx3d[0] * cc_block_shape[0], (idx3d[0] + 1) * cc_block_shape[0]),
                slice(idx3d[1] * cc_block_shape[1], (idx3d[1] + 1) * cc_block_shape[1]),
                slice(idx3d[2] * cc_block_shape[2], (idx3d[2] + 1) * cc_block_shape[2]),
            )
            cur_fname = f"{fname_prefix}.{blockid}"
            if blockid == 0:
                root_fname = cur_fname
            with h5py.File(cur_fname, "w") as f:
                f.attrs["dims"] = np.array(global_arrays[field_names[0]].shape)
                if concat_format is not ConcatFormat.Legacy:
                    f.attrs["offset"] = np.array([int(slc.start) for slc in layout_slc])
                    f.attrs["dims_local"] = cc_block_shape
                f.attrs["nprocs"] = np.array(nprocs)
                for field_name in field_names:
                    f[field_name] = global_arrays[field_name][layout_slc]
        return root_fname

    if concat_format is ConcatFormat.Modern:
        # this is a really dumb way to do things, but it gets the job done until we
        # have time to implement something smarter
        out_dir = os.path.dirname(root_path)
        script = os.path.join(_PYTHON_SCRIPTS_PATH, "concat_3d_data.py")

        python_binary = sys.executable
        with TemporaryDirectory() as tmpdirname:
            _work(fname_prefix=os.path.join(tmpdirname, "0.h5"))
            args = [
                "--source-directory",
                tmpdirname,
                "--output-directory",
                out_dir,
                "--snaps",
                "0",
            ]
            subprocess.run(
                [python_binary, script] + args, check=True, capture_output=True
            )
        cur_fname = os.path.join(out_dir, "0.h5")
        root_fname = f"{root_path}.h5"
        if os.path.normpath(cur_fname) != os.path.normpath(root_fname):
            os.rename(cur_fname, root_fname)
    else:
        root_fname = _work(fname_prefix=f"{root_path}.h5")
    return global_arrays, root_fname


_CASES = [
    {"nprocs": (1, 1, 1), "global_shape": (8, 8, 8), "concat_format": None},
    {"nprocs": (2, 2, 2), "global_shape": (4, 16, 8), "concat_format": None},
    {"nprocs": (1, 4, 2), "global_shape": (4, 16, 8), "concat_format": None},
    {
        "nprocs": (2, 2, 2),
        "global_shape": (4, 16, 8),
        "concat_format": ConcatFormat.Legacy,
    },
    {
        "nprocs": (2, 2, 2),
        "global_shape": (4, 16, 8),
        "concat_format": ConcatFormat.Modern,
    },
]


@pytest.mark.parametrize("kwargs", _CASES)
def test_load(tmp_path, kwargs):
    # this scenario imagines we ran a simulation with a single process that output a
    # single file
    field_names = ["my-field", "my-field2"]

    global_arr, root_fname = _generate_files(
        str(tmp_path / "dummy"), field_names=field_names, **kwargs
    )

    loaded = cholla_utils.load_field(root_fname, field=field_names[-1])
    np.testing.assert_equal(global_arr[field_names[-1]], loaded)

    loaded = cholla_utils.load_field(root_fname, field=field_names)
    for name in field_names:
        np.testing.assert_equal(global_arr[name], loaded[name])


@pytest.mark.parametrize("kwargs", _CASES)
def test_subarray(tmp_path, kwargs):
    # this scenario imagines we ran a simulation with a single process that output a
    # single file
    field_names = ["my-field", "my-field2"]

    global_arr, root_fname = _generate_files(
        str(tmp_path / "dummy"), field_names=field_names, **kwargs
    )

    idx = np.s_[1, 2:, -3:-1]

    loaded = cholla_utils.load_field(root_fname, field=field_names[-1], idx=idx)
    np.testing.assert_equal(global_arr[field_names[-1]][idx], loaded)

    loaded = cholla_utils.load_field(root_fname, field=field_names, idx=idx)
    for name in field_names:
        np.testing.assert_equal(global_arr[name][idx], loaded[name])


def test_simple_failure(tmp_path):
    field_name = "my-field"

    global_arrs, root_fname = _generate_files(
        str(tmp_path / "dummy"),
        nprocs=[1, 1, 1],
        global_shape=(4, 16, 8),
        field_names=[field_name],
    )

    with pytest.raises(KeyError):
        cholla_utils.load_field(root_fname, field="not-a-field")

    with pytest.raises(FileNotFoundError):
        cholla_utils.load_field("/not/a/file", field="not-a-field")
