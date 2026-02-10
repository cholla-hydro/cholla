"""
Define some utilities shared among tests.
"""

import pathlib
import pytest

from cholla_utils._testing import (
    create_snap_from_input_pack,
    SyntheticInputPack,
    SyntheticSnapResult,
)


@pytest.fixture(scope="function")
def synth_snap(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> SyntheticSnapResult:
    """
    Creates & writes a synthetic snapshot to disk & returns a summary

    Cleanup is controlled through pytest's built-in tmp_path fixture.
    """
    # arg should be a SyntheticInputPack instance
    arg = request.param
    if not isinstance(arg, SyntheticInputPack):
        raise RuntimeError("synthetic_snap fixture recieved an arg of the wrong type")
    out = create_snap_from_input_pack(str(tmp_path / "0"), input_pack=arg)

    # sanity check
    if (out.field_data is None) != (out.root_field_fname is None):
        raise RuntimeError("something went wrong with field data")
    elif (out.particle_data is None) != (out.root_particle_fname is None):
        raise RuntimeError("something went wrong with particle data")
    return out


def pytest_make_parametrize_id(config, val):
    # this is a hook function to customize the ids that pytest infers when
    # functions are parametrized
    if isinstance(val, SyntheticInputPack):
        return repr(val)
