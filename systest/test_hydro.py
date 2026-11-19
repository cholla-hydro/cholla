from dataclasses import dataclass
import os
import pathlib

import cholla_utils
import numpy as np
import pytest

from utils import (
    SimTester,
    assert_expected_uniform_vals,
    converge_error_test,
    ClassicAbsErrReqs,
    perform_answer_test,
)

# TODO: come up with a better solution (for this stuff)
_TMP = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = _TMP / "src" / "system_tests" / "input_files"


def _legacy_fname(prefix: str) -> pathlib.Path:
    return _ROOT / f"{prefix}CorrectInputExpectCorrectOutput.txt"


# @pytest.mark.bpreset_mhd
@pytest.mark.bpreset_hydro
def test_sod3d(simtester, answertestspec):
    _LEGACY_TEST_NAME = (
        "tHYDROtMHDSYSTEMSodShockTubeParameterizedMpi_CorrectInputExpectCorrectOutput"
    )

    build_preset = simtester.binary_props["build_preset"]
    if build_preset == "mhd":
        pytest.skip("logic needs to be ported over")

    rslt = simtester.exec_single_sim(
        nproc=4, param_file=_ROOT / (_LEGACY_TEST_NAME + ".txt")
    )

    perform_answer_test(
        actual_path=rslt.path / "1" / "1.h5.0",
        ref_path=answertestspec.answer_dir / (_LEGACY_TEST_NAME + ".h5"),
        expected_step_count=None,
        err_spec=ClassicAbsErrReqs(0.0, 0),
    )


@pytest.mark.bpreset_hydro
def test_sod1d(simtester, answertestspec):
    _LEGACY_TEST_NAME = (
        "tHYDROSYSTEMSodShockTube_OneDimensionalCorrectInputExpectCorrectOutput"
    )
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_ROOT / (_LEGACY_TEST_NAME + ".txt")
    )

    perform_answer_test(
        actual_path=rslt.path / "1" / "1.h5.0",
        ref_path=answertestspec.answer_dir / (_LEGACY_TEST_NAME + ".h5"),
        expected_step_count=None,
        err_spec=ClassicAbsErrReqs(),
    )


@pytest.mark.bpreset_hydro
def test_sod2d(simtester, answertestspec):
    _LEGACY_TEST_NAME = (
        "tHYDROSYSTEMSodShockTube_TwoDimensionalCorrectInputExpectCorrectOutput"
    )
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_ROOT / (_LEGACY_TEST_NAME + ".txt")
    )

    perform_answer_test(
        actual_path=rslt.path / "1" / "1.h5.0",
        ref_path=answertestspec.answer_dir / (_LEGACY_TEST_NAME + ".h5"),
        expected_step_count=None,
        err_spec=ClassicAbsErrReqs(),
    )


@pytest.mark.bpreset_hydro
@pytest.mark.bpreset_mhd
def test_simple(simtester: SimTester):
    # run the sim
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_legacy_fname("tHYDROtMHDSYSTEMConstant_")
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


@pytest.mark.bpreset_hydro
@pytest.mark.bpreset_mhd
def test_sound3d(simtester: SimTester):
    # run the sim
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_legacy_fname("tHYDROtMHDSYSTEMSoundWave3D_")
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


@dataclass
class LinWaveErr:
    max_L1_errvec_mag: float
    max_Linf_errvec_comp: float


# these all come from the historical C++ system-testing framework
# -> in the old code, each line defining a value of (what we now call) max_L1_errvec_mag
#    had a trailing comment "Based on results in Gardiner & Stone 2008"
# -> I'm honestly not sure we should base these error thresholds off of those papers
# -> it would be more robust for us to set errors for particular kinds of waves
_lin_wave_err_requirements = {
    "PCM": LinWaveErr(4.0e-7, 4.0e-7),
    "PLMC": LinWaveErr(1.0e-7, 1.0e-7),
    "PLMP": LinWaveErr(1.0e-7, 1.0e-7),
    "PPMC": LinWaveErr(2.7e-8, 2.7e-8),
    "PPMP": LinWaveErr(2.7e-8, 2.7e-8),
}


def _lin_wave_launch_args(
    wave_speed: float,
    rEigenVec_rho: float,
    rEigenVec_MomentumX: float,
    rEigenVec_MomentumY: float,
    rEigenVec_MomentumZ: float,
    rEigenVec_E: float,
    vx: float = 0.0,
) -> dict[str, float | str]:
    N = 32
    domain = 0.5
    gamma = 5.0 / 3.0
    tOut = 2 * domain / wave_speed

    return {
        "nx": 2 * N,
        "ny": N,
        "nz": N,
        "tout": tOut,
        "outstep": tOut,
        "init": "Linear_Wave",
        "xmin": 0.0,
        "ymin": 0.0,
        "zmin": 0.0,
        "xlen": 2 * domain,
        "ylen": domain,
        "zlen": domain,
        "xl_bcnd": 1,
        "xu_bcnd": 1,
        "yl_bcnd": 1,
        "yu_bcnd": 1,
        "zl_bcnd": 1,
        "zu_bcnd": 1,
        "rho": 1.0,
        "vx": vx,
        "vy": 0.0,
        "vz": 0.0,
        "P": 1 / gamma,
        "Bx": 0.0,
        "By": 0.0,
        "Bz": 0.0,
        "A": 1e-6,
        "gamma": gamma,
        "rEigenVec_rho": rEigenVec_rho,
        "rEigenVec_MomentumX": rEigenVec_MomentumX,
        "rEigenVec_MomentumY": rEigenVec_MomentumY,
        "rEigenVec_MomentumZ": rEigenVec_MomentumZ,
        "rEigenVec_E": rEigenVec_E,
        "rEigenVec_Bx": 0.0,
        "rEigenVec_By": 0.0,
        "rEigenVec_Bz": 0.0,
    }


def test_sound_right_moving(simtester: SimTester):
    reconstruct_kind = simtester.binary_props["reconstruct_kind"]
    err_req = _lin_wave_err_requirements[reconstruct_kind]

    launch_args = _lin_wave_launch_args(
        wave_speed=1.0,
        rEigenVec_rho=1,
        rEigenVec_MomentumX=1,
        rEigenVec_MomentumY=1,
        rEigenVec_MomentumZ=1,
        rEigenVec_E=1.5,
    )

    rslt = simtester.exec_single_sim(
        nproc=1, param_file=None, override_params=launch_args
    )

    converge_error_test(
        rslt.path / "1.h5.0",
        rslt.path / "0.h5.0",
        err_msg="",
        max_L1_errvec_mag=2 * err_req.max_L1_errvec_mag,
        max_Linf_errvec_comp=err_req.max_L1_errvec_mag,
        expected_step_count=214,
        is_initial_final=True,
    )


@pytest.mark.bpreset_hydro
@pytest.mark.bpreset_mhd
def test_sound_left_moving(simtester: SimTester):
    reconstruct_kind = simtester.binary_props["reconstruct_kind"]
    err_req = _lin_wave_err_requirements[reconstruct_kind]

    launch_args = _lin_wave_launch_args(
        wave_speed=1.0,
        rEigenVec_rho=1,
        rEigenVec_MomentumX=-1,
        rEigenVec_MomentumY=1,
        rEigenVec_MomentumZ=1,
        rEigenVec_E=1.5,
    )

    rslt = simtester.exec_single_sim(
        nproc=1, param_file=None, override_params=launch_args
    )

    converge_error_test(
        rslt.path / "1.h5.0",
        rslt.path / "0.h5.0",
        err_msg="",
        max_L1_errvec_mag=2 * err_req.max_L1_errvec_mag,
        max_Linf_errvec_comp=err_req.max_L1_errvec_mag,
        expected_step_count=214,
        is_initial_final=True,
    )


@pytest.mark.bpreset_hydro
@pytest.mark.bpreset_mhd
def test_contact_wave(simtester: SimTester):
    reconstruct_kind = simtester.binary_props["reconstruct_kind"]
    err_req = _lin_wave_err_requirements[reconstruct_kind]

    wave_speed = 1.0
    launch_args = _lin_wave_launch_args(
        wave_speed=wave_speed,
        rEigenVec_rho=1,
        rEigenVec_MomentumX=-1,
        rEigenVec_MomentumY=0,
        rEigenVec_MomentumZ=0,
        rEigenVec_E=0.5,
        vx=wave_speed,
    )

    rslt = simtester.exec_single_sim(
        nproc=1, param_file=None, override_params=launch_args
    )

    converge_error_test(
        rslt.path / "1.h5.0",
        rslt.path / "0.h5.0",
        err_msg="",
        max_L1_errvec_mag=err_req.max_L1_errvec_mag,
        max_Linf_errvec_comp=err_req.max_L1_errvec_mag,
        expected_step_count=427,
        is_initial_final=True,
    )


@pytest.mark.bpreset_hydro
def test_kh(simtester, answertestspec):
    _LEGACY_TEST_NAME = "tHYDROSYSTEMKhDiscontinuous_CorrectInputExpectCorrectOutput"
    rslt = simtester.exec_single_sim(
        nproc=1, param_file=_ROOT / (_LEGACY_TEST_NAME + ".txt")
    )

    perform_answer_test(
        actual_path=rslt.path / "1" / "1.h5.0",
        ref_path=answertestspec.answer_dir / (_LEGACY_TEST_NAME + ".h5"),
        expected_step_count=None,
        err_spec=ClassicAbsErrReqs(),
    )
