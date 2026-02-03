import os
import subprocess
import sys

import pytest

import cholla_utils

from cholla_utils._misc import ChollaDataFmt
from cholla_utils._testing import (
    BlockPreset,
    assert_arraydict_equal,
    USE_DEFAULT,
    SyntheticInputPack,
)

_PYTHON_SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _call_concat(out_dir, src_root_fname):
    src_dir, src_basename = os.path.split(src_root_fname)
    if src_basename != "0.h5.0":
        raise ValueError("an implicit assumption about input data is wrong")
    out_root_fname = os.path.join(out_dir, "0.h5")

    os.mkdir(out_dir)

    # actually perform concatenation
    cmd = [
        sys.executable,  # <- python binary
        os.path.join(_PYTHON_SCRIPTS_PATH, "concat_3d_data.py"),
        "--source-directory",
        src_dir,
        "--output-directory",
        out_dir,
        "--snaps",
        "0",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_root_fname


_SINGLEBLOCK_CASE = SyntheticInputPack(
    blocking_preset=BlockPreset.CubeDomain_1,
    field=USE_DEFAULT,
    particle=None,
    datafmt=ChollaDataFmt.DISTRIBUTED,
)


@pytest.mark.parametrize("synth_snap", [_SINGLEBLOCK_CASE], indirect=True)
def test_field_concat_fail(synth_snap, tmp_path):
    out_dir = tmp_path / "my_concat"

    # this fails since the dataset is already a single block
    # -> we may want to revisit this choice in the future
    with pytest.raises(subprocess.CalledProcessError):
        _call_concat(out_dir, synth_snap.root_field_fname)


_CASES = [
    SyntheticInputPack(
        blocking_preset=blocking_preset,
        field=USE_DEFAULT,
        particle=None,
        datafmt=ChollaDataFmt.DISTRIBUTED,
    )
    for blocking_preset in [BlockPreset.IrrDomain_142, BlockPreset.IrrDomain_222]
]


@pytest.mark.parametrize("synth_snap", _CASES, indirect=True)
def test_field_concat(synth_snap, tmp_path):
    out_dir = tmp_path / "my_concat"
    root_fname = _call_concat(out_dir, synth_snap.root_field_fname)

    # here, we assume are assuming that the cholla_utils functions all work correctly
    global_arr = synth_snap.field_data
    loaded = cholla_utils.load_field(root_fname, field=list(global_arr))
    assert_arraydict_equal(
        loaded, global_arr, err_msg="comparing fields loaded together"
    )
