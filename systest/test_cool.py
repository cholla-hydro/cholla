import os
import pathlib

from utils import SimTester, assert_expected_uniform_vals

# TODO: come up with a better solution (for this stuff)
_TMP = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = _TMP / "src" / "system_tests" / "input_files"


def _legacy_fname(prefix: str) -> pathlib.Path:
    return _ROOT / f"{prefix}_CorrectInputExpectCorrectOutput.txt"


_COOL_RHO = 6.9498489284711


def test_cool5(simtester: SimTester):
    # comment from c++
    # ----------------
    ### dt = 0.3
    ### rho = COOL_RHO*1e5
    ### pressure = 1e-3
    ### T = 1e5
    ### energy = 0.0014850544057189395 # Python
    # energy = 0.00148501098087863  # Cholla

    # run the sim
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_legacy_fname("tHYDROtMHDSYSTEMCoolingConstant5")
    )

    # actually perform the check
    assert_expected_uniform_vals(
        path=rslt.path / "1" / "1.h5.0",
        expected={
            "density": _COOL_RHO * 1e5,
            "momentum_x": 0.0,
            "momentum_y": 0.0,
            "momentum_z": 0.0,
            "Energy": 0.00148501098087863,
        },
    )


def test_cool7(simtester: SimTester):
    # comment from c++
    # ----------------
    ### dt = 100
    ### rho = COOL_RHO*1e5
    ### pressure = 1e-1
    ### T = 1e7
    ### energy = 0.14982743570299709 # Python
    # energy = 0.14982745510047499  # Cholla

    # run the sim
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_legacy_fname("tHYDROtMHDSYSTEMCoolingConstant7")
    )

    # actually perform the check
    assert_expected_uniform_vals(
        path=rslt.path / "1" / "1.h5.0",
        expected={
            "density": _COOL_RHO * 1e5,
            "momentum_x": 0.0,
            "momentum_y": 0.0,
            "momentum_z": 0.0,
            "Energy": 0.14982745510047499,
        },
    )


def test_cool8(simtester: SimTester):
    # comment from c++
    # ----------------
    ### dt = 90
    ### rho = COOL_RHO*1e5
    ### pressure = 1
    ### T = 1e8
    ### energy = 1.499669522009355 # Python
    # energy = 1.4996695198095711 # Cholla

    # run the sim
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_legacy_fname("tHYDROtMHDSYSTEMCoolingConstant8")
    )

    # actually perform the check
    assert_expected_uniform_vals(
        path=rslt.path / "1" / "1.h5.0",
        expected={
            "density": _COOL_RHO * 1e5,
            "momentum_x": 0.0,
            "momentum_y": 0.0,
            "momentum_z": 0.0,
            "Energy": 1.4996695198095711,
        },
    )
