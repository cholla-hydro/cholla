import dataclasses
import os
import pathlib
import re
import shutil
import subprocess
import types
import time

import cholla_utils
import numpy as np
import pytest


class ChollaSpec:
    """Encodes information about cholla and how to construct a command"""

    _common_args: list[str]

    def __init__(self, launcher: str, cholla_bin: str):
        # build up common_args
        self._common_args = []
        launcher = [] if launcher is None else launcher.split(";")
        if len(launcher) > 0:
            if "{nproc}" in launcher[0]:
                raise ValueError("{nproc} can't appear in first arg of launcher seq")
            elif shutil.which(launcher[0]) is None:
                raise ValueError(
                    f"can't find executable associated with `{launcher[0]}`"
                )
            self._common_args.append(launcher[0])
        for i, elem in enumerate(launcher[1:], start=1):
            try:
                elem.format(nproc=2)
            except KeyError:
                raise ValueError(
                    f"launcher cmd component `{elem}` is invalid"
                ) from None
            self._common_args.append(elem)
        if not os.path.isfile(cholla_bin):
            raise ValueError(f"{cholla_bin} is not a valid path")
        self._common_args.append(os.path.abspath(cholla_bin))

    @property
    def compile_profile(self) -> str:
        # basic premise: this will eventually help us identify compatible tests
        # -> we'll dynamically set this from the command line
        # -> we'll add marks to the tests to denote compatability
        return "hydro"

    def build_cmd(self, nproc: int, param_file: str, **override_params) -> list[str]:
        """Build a command that can be passed to subprocess.run"""
        out = [self._common_args[0]]
        for elem in self._common_args[1:-1]:
            out.append(elem.format(nproc=nproc))
        if len(self._common_args) > 1:
            out.append(self._common_args[-1])
        out.append(param_file)
        # todo: we probably need to be a little more careful about escaping string vals
        out.extend(f"{k}={v!r}" for k, v in override_params.items())
        return out


def test_name_to_basename(name: str):
    # coerce a pytest name to the name of a directory
    # -> we don't want this level to imply that there are multiple directory levels
    # -> a pytest-name can include all the components of a file-path & everything in a
    #    python identfier
    # -> can also include anything related to parametrize

    # this is probably overly-restrictive
    return re.sub("[^-,_0-9A-Za-z~]", "_", name)


@dataclasses.dataclass(frozen=True)
class SimConf:
    """Configuration of a single simulation"""

    nproc: int
    param_file: str
    override_params: dict[str, int | float | str | bool]
    timeout: float | None


@dataclasses.dataclass(frozen=True)
class SimResult:
    """Encodes the result of a simulation execution"""

    # whether the simulation succeeded
    success: bool
    # path to the directory containing the result
    path: pathlib.Path
    # the duration of the simulation in nanoseconds
    # (ugh, this has some annoying implications for reuse)
    duration_ns: int | None
    # we can add other stuff like stdout and stderr in the future
    # -> they have some annoying implications for reuse...
    # -> maybe we only support access to these when we expect a failure?


class SimTester:
    """Provides the machinery to run a system-test of a simulation code

    This is very much inspired by the Pytester object provided inside of pytest

    HIGH PRIORITY: make it possible to reuse outputs from a previous simulation run
    (if we don't bother supporting this, then I think we should pick a totally
    different design...)
    """

    # all simulation-related artifacts for current test stored in test-dir
    _test_dir: pathlib.Path
    # specifies information about building a command
    _sim_spec: ChollaSpec
    # used to track all sim-configs
    _sim_confs: dict[str, SimConf]
    # tracks whether the test has been invoked
    _already_executed: bool
    # whether to reuse existing output
    _reuse_existing: bool

    def __init__(
        self, test_dir: pathlib.Path, sim_spec: ChollaSpec, reuse_existing: bool = False
    ):
        self._test_dir = test_dir
        self._sim_spec = sim_spec
        self._sim_confs = {}
        self._already_executed = False
        self._reuse_existing = reuse_existing

    @property
    def configs(self) -> types.MappingProxyType[str, SimConf]:
        return types.MappingProxyType(self._sim_confs)

    def configure_sim(
        self,
        name: str,
        nproc: int,
        param_file: str | pathlib.Path,
        *,
        override_params: dict[str, int | float | str | bool] | None = None,
        set_indir_from_name_output: str | None = None,
        timeout: float | None = None,
    ):
        """Adds a new simulation configuration"""
        # Future Additions:
        # - add a kwarg to make it possible to use the outdir of a previously specified
        #   simulation configuration
        # - maybe add a kwarg to allow an expected failure?
        if self._already_executed:
            raise ValueError("this method can't be invoked after calling exec")
        elif name in self._sim_confs:
            raise ValueError(f"a simulation config called {name!r} already registered")
        elif name != test_name_to_basename(name):
            raise ValueError("invalide name")

        override_params = {} if override_params is None else override_params

        # here we wire things up so that we properly restart from an output of a sim
        # that will also be run by the SimTester
        if set_indir_from_name_output is not None:
            if "indir" in override_params:
                raise ValueError("indir parameter is being overridden twice")
            try:
                referenced = self._sim_confs[set_indir_from_name_output]
            except KeyError:
                raise ValueError(
                    f"{set_indir_from_name_output!r} is not an already registered sim"
                )
            tmp = (
                self._test_dir
                / set_indir_from_name_output
                / referenced.override_params.get("outdir", ".")
            )
            # this is a bit of a hack!
            if "nfile" in override_params:
                tmp = str(tmp / str(override_params["nfile"])) + "/"
            else:
                tmp = str(tmp)
            override_params["indir"] = str(tmp)

        self._sim_confs[name] = SimConf(
            nproc=nproc,
            param_file=os.path.abspath(param_file),
            override_params=override_params,
            timeout=timeout,
        )

    def exec(self) -> dict[str, SimResult]:
        """Execute each simulation configuration"""
        __tracebackhide__ = True  # <- # Hide traceback for py.test

        if self._already_executed:
            raise ValueError("exec can't be called more than once")
        elif len(self._sim_confs) == 0:
            raise ValueError("no simulations were configured")

        if self._reuse_existing:
            raise NotImplementedError("reuse-mode isn't supported yet!")

        out = {}
        for name, config in self._sim_confs.items():
            out_dir = self._test_dir / name
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir()

            kw = {} if config.override_params is None else config.override_params
            args = self._sim_spec.build_cmd(
                nproc=config.nproc, param_file=config.param_file, **kw
            )
            # todo: we should really log both:
            # -> the name of the simulation
            # -> the command being executed
            print(f"executing:\n  {' '.join(args)}")

            # todo: figure out what do with:
            # - we probably want to be checking the output exit code
            # - stdout and stderr (we probably want to capture it)
            #
            # keep in mind that this has implications for reusing outputs!
            t1 = time.perf_counter_ns()
            tmp_rslt = subprocess.run(args, timeout=config.timeout, cwd=out_dir)
            t2 = time.perf_counter_ns()
            if tmp_rslt.returncode != 0:
                pytest.fail(
                    f"failure when running:\n  {' '.join(args)}\nfrom {out_dir}"
                )
            out[name] = SimResult(success=True, path=out_dir, duration_ns=t2 - t1)
        return out

    def exec_single_sim(
        self,
        nproc: int,
        param_file: str | pathlib.Path,
        *,
        override_params: dict[str, int | float | str | bool] | None = None,
        timeout: float | None = None,
    ) -> SimResult:
        """A convenience method for when a SimTester is only used to run 1 simulation

        This is equivalent to calling `obj.configure_sim(name, ...)` and then calling
        `obj.exec()[name]`.
        """
        if len(self._sim_confs) > 0:
            raise ValueError(
                "no simulations can be configured before calling this method"
            )
        self.configure_sim(
            "single-sim",
            nproc=nproc,
            param_file=param_file,
            override_params=override_params,
            timeout=timeout,
        )
        return self.exec()["single-sim"]


# Custom assertions
# -----------------


def assert_expected_uniform_vals(
    path: os.PathLike,
    expected: dict[str, float],
    *,
    atol: float = 5.0e-12,
    rtol: float = 0.0,
):
    """Checks that the snapshot at the specified path has unified values

    The defaul atol and rtol values are based on the historical googletest
    """
    __tracebackhide__ = True  # <- # Hide traceback for py.test

    path = pathlib.Path(path)
    if not path.exists():
        pytest.fail(f"no file at: {path}")

    actual_field_list = cholla_utils.get_native_fields(path)
    assert set(actual_field_list) == set(expected.keys()), "unexpected field list"
    data = cholla_utils.load_field(path, actual_field_list)
    for field, val in expected.items():
        np.testing.assert_allclose(
            data[field],
            desired=val,
            err_msg=f"unexpected {field!r} value in {path}",
            atol=atol,
            rtol=rtol,
        )
