# this defines some basic utilities shared among all of the tests

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys

import pytest

from utils import ChollaSpec, SimTester, test_name_to_basename


@dataclasses.dataclass(frozen=True)
class AnswerTestSpec:
    generate_answers: bool
    answer_dir: pathlib.Path | None


_ANSWER_TEST_SPEC_KEY = pytest.StashKey[AnswerTestSpec]()
_CHOLLA_SPEC_KEY = pytest.StashKey[ChollaSpec]()

_KNOWN_BUILD_PRESETS = set(["bpreset_hydro", "bpreset_mhd", "bpreset_cosmology"])


# this hook is used to add more command line flags to the pytest launcher
def pytest_addoption(parser):
    parser.addoption(
        "--cholla-bin", action="store", required=True, help="Path to the cholla binary."
    )
    # ugh, unfortunately the following breaks with nargs=*
    parser.addoption(
        "--launcher",
        action="store",
        help=(
            "semicolon delimited list of launcher arguments. Occurrences of {nproc} "
            "will be replaced wth the desired number of process"
        ),
    )
    # in the future, it would be nice if the following were runtime-choices or could at
    # least be inferred from the binary
    parser.addoption(
        "--build-preset",
        choices=("hydro", "mhd", "cosmology"),
        default=None,
        help="the build-preset of the cholla binary",
    )
    parser.addoption(
        "--reconstruct-kind",
        choices=("PCM", "PLMC", "PLMP", "PPMC", "PPMP"),
        default=None,
        help="the reconstruction strategy hardcoded in the cholla binary",
    )

    parser.addoption(
        "--answer-dir",
        action="store",
        default=None,
        help=(
            "Path to directory used for storing answer-tests answers. All "
            "logic related to answer-tests is disabled unless this is "
            "specified. When specified alonside --answer-store, logic to "
            "generate the answers will be run (the answers are stored to this "
            "directory). When specified WITHOUT --answer-store, the results "
            "of each answer-test is compared against the answers that were "
            "previously stored in this directory."
        ),
    )

    parser.addoption(
        "--answer-store",
        action="store_true",
        help=(
            "Indicates that we should store answer-test results. It is an "
            "error to specifiy this arg without --answer-dir."
        ),
    )


# this hook gets executed after pytest finishes parsing all of the config opts
def pytest_configure(config):
    for elem in _KNOWN_BUILD_PRESETS:
        config.addinivalue_line("markers", f"{elem}: a build preset")

    # this is a hack!
    if any("--help" == arg for arg in sys.argv[1:]):
        return

    # handle ChollaSpec
    # -----------------
    if config.getoption("cholla_bin") is None:
        raise pytest.UsageError("--cholla-bin is missing")
    elif config.getoption("build_preset") is None:
        raise pytest.UsageError("--build-preset is missing")
    elif config.getoption("reconstruct_kind") is None:
        raise pytest.UsageError("--reconstruct-kind is missing")

    config.stash[_CHOLLA_SPEC_KEY] = ChollaSpec(
        launcher=config.getoption("launcher"),
        cholla_bin=config.getoption("cholla_bin"),
        build_preset=config.getoption("build_preset"),
        reconstruct_kind=config.getoption("reconstruct_kind"),
    )

    # assemble AnswerTestSpec
    # -----------------------
    if config.getoption("--answer-store"):
        raise pytest.UsageError("--answer-store option isn't fully enabled yet!")
    generate_answers = config.getoption("--answer-store")
    answer_dir = config.getoption("--answer-dir")

    if (answer_dir is None) and generate_answers:
        raise pytest.UsageError(
            "--answer-store option can't be specified without --answer-dir"
        )
    elif answer_dir is not None:
        if (not os.path.isdir(answer_dir)) and (not generate_answers):
            raise pytest.UsageError(
                f"directory of test answers can't be found: {answer_dir}"
            )
        elif generate_answers:
            os.makedirs(answer_dir, exist_ok=True)
        answer_dir = pathlib.Path(answer_dir)

    config.stash[_ANSWER_TEST_SPEC_KEY] = AnswerTestSpec(
        generate_answers=generate_answers, answer_dir=answer_dir
    )


# hook for printing information at the top of the test report
def pytest_report_header(config):
    return [
        f"cholla-bin: {config.getoption('cholla_bin')}",
        f"launcher: {config.getoption('launcher')}",
        f"answer-dir: {config.getoption('--answer-dir')}",
    ]


@pytest.fixture
def simtester(request: pytest.FixtureRequest) -> SimTester:
    # todo: make root configurable (can we use tmp_path_factory?)
    root = pathlib.Path.cwd() / "test-work-dir"
    root.mkdir(exist_ok=True)

    # get test_dir: it contains all simulation-related artifacts for current test
    test_name = request.node.name
    test_dir = root / test_name_to_basename(test_name)
    test_dir.mkdir(exist_ok=True)

    return SimTester(test_dir=test_dir, sim_spec=request.config.stash[_CHOLLA_SPEC_KEY])


@pytest.fixture(scope="session")
def answertestspec(request: pytest.FixtureRequest) -> SimTester:
    # todo: most answer-testing logic should be performed within a pytest_runtest_call
    #       hook function
    obj = request.config.stash[_ANSWER_TEST_SPEC_KEY]
    if obj.answer_dir is None:
        pytest.skip("no --answer-dir option specified")
    return obj


def pytest_runtest_setup(item):
    supported_build_preset = _KNOWN_BUILD_PRESETS.intersection(
        mark.name for mark in item.iter_markers()
    )

    build_preset = (
        "bpreset_" + item.config.stash[_CHOLLA_SPEC_KEY].props["build_preset"]
    )
    if supported_build_preset and build_preset not in supported_build_preset:
        pytest.skip(f"cannot run for build preset: {build_preset}")
