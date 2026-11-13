import os
import pathlib

import cholla_utils
import numpy as np

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


def _mk_analytic_sine(
    constant: float,
    amplitude: float,
    phase: float,
    k_vec: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> np.ndarray:
    kx, ky, kz = k_vec
    coord_args = [np.arange(extent) for extent in shape]
    xv, yv, zv = np.meshgrid(*coord_args, indexing="ij", sparse=True)
    return constant + amplitude * np.sin(xv * kx + yv * ky + zv * kz + phase)


def test_sound3d(simtester: SimTester):
    # run the sim
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_legacy_fname("tHYDROtMHDSYSTEMSoundWave3D")
    )

    build_preset = simtester.binary_props["build_preset"]
    recon = simtester.binary_props["reconstruct_kind"]

    if build_preset == "hydro":
        tol = 1.0e-7
    elif build_preset == "mhd":
        # Loosen correctness check to account for MHD only having PCM. This is
        # about the error between PCM and PPMP in hydro
        # Check Results. Values based on results in Gardiner & Stone 2008
        if recon == "PCM":
            tol = 1e-6
        elif recon == "PLMC":
            tol = 1.0e-7
        elif recon == "PPMC":
            tol = 1.9e-9
        else:
            raise RuntimeError(f"unexpected reconstruction method: {recon}")
    else:
        raise RuntimeError()

    # load the result
    data = cholla_utils.load_field(
        rslt.path / "1" / "1.h5.0", ["density", "momentum_x"]
    )

    # do the comparison
    time = 0.05
    dx = 1.0 / 64

    real_kx = 2 * np.pi  # kx of the physical problem

    kx = real_kx * dx
    speed = 1.0  # speed of wave is 1 since P = 0.6 and gamma = 1.666667
    phase = kx * 0.5 - speed * time * real_kx  # kx*0.5 for half-cell offset

    common_kwargs = {
        "amplitude": 1.0e-5,
        "phase": phase,
        "k_vec": (kx, 0.0, 0.0),
        "shape": data["density"].shape,
    }

    np.testing.assert_allclose(
        data["density"],
        _mk_analytic_sine(constant=1.0, **common_kwargs),
        err_msg='unexpected "density" field value',
        rtol=0.0,
        atol=tol,
    )

    np.testing.assert_allclose(
        data["momentum_x"],
        _mk_analytic_sine(constant=0.0, **common_kwargs),
        err_msg='unexpected "density" field value',
        rtol=0.0,
        atol=tol,
    )
