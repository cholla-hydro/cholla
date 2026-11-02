import os
import pathlib

from utils import SimTester, assert_expected_uniform_vals

# TODO: come up with a better solution (for this stuff)
_TMP = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = _TMP / "src" / "system_tests" / "input_files"


def _legacy_fname(prefix: str) -> pathlib.Path:
    return _ROOT / f"{prefix}_CorrectInputExpectCorrectOutput.txt"


def test_simple(simtester: SimTester):
    # run the sim
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_legacy_fname("tHYDROtMHDSYSTEMConstant")
    )

    # actually perform the check
    assert_expected_uniform_vals(
        path=rslt.path / "1" / "1.h5.0",
        expected={
            "density": 1.0,
            "momentum_x": 0.0,
            "momentum_y": 0.0,
            "momentum_z": 0.0,
            "Energy": 1.5e-5,
        },
    )
