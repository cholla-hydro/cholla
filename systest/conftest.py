# this defines some basic utilities shared among all of the tests

from __future__ import annotations

import pathlib
import sys

import pytest

from utils import ChollaSpec, SimTester, test_name_to_basename

_CHOLLA_SPEC_KEY = pytest.StashKey[ChollaSpec]()


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


# this hook gets executed after pytest finishes parsing all of the config opts
def pytest_configure(config):
    # this is a hack!
    if any("--help" == arg for arg in sys.argv[1:]):
        return

    config.stash[_CHOLLA_SPEC_KEY] = ChollaSpec(
        launcher=config.getoption("launcher"), cholla_bin=config.getoption("cholla_bin")
    )


# hook for printing information at the top of the test report
def pytest_report_header(config):
    return [
        f"cholla-bin: {config.getoption('cholla_bin')}",
        f"launcher: {config.getoption('launcher')}",
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
