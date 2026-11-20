from collections.abc import Collection, Iterator, Sequence
import dataclasses
import math
import os
import pathlib
import re
import shutil
import subprocess
from types import MappingProxyType
import time

import cholla_utils
import h5py
import numpy as np
import pytest


class ChollaSpec:
    """Encodes information about cholla and how to construct a command"""

    _common_args: list[str]
    props: MappingProxyType[str, str]

    def __init__(
        self,
        launcher: str | None,
        cholla_bin: str,
        build_preset: str,
        reconstruct_kind: str,
    ):
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
        self.props = MappingProxyType(
            {"build_preset": build_preset, "reconstruct_kind": reconstruct_kind}
        )

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
    def binary_props(self) -> MappingProxyType[str, str]:
        return self._sim_spec.props

    @property
    def configs(self) -> MappingProxyType[str, SimConf]:
        return MappingProxyType(self._sim_confs)

    def configure_sim(
        self,
        name: str,
        nproc: int,
        param_file: str | pathlib.Path | None,
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

        if param_file is None:
            if len(override_params) == 0:
                raise ValueError(
                    "param_file can only be None if override_params are provided"
                )
            systest_root = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
            param_file = systest_root / "input" / "blank_settings_file.txt"

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
            log_file_name = out_dir / "sim.log"
            print(f"executing:\n  {' '.join(args)}\nlogging to: {log_file_name}")

            error = None
            with open(log_file_name, "w") as log_file:
                try:
                    t1 = time.perf_counter_ns()
                    tmp_rslt = subprocess.run(
                        args,
                        timeout=config.timeout,
                        cwd=out_dir,
                        # combine stdout and stderr and write to log_file
                        stderr=subprocess.STDOUT,
                        stdout=log_file,
                    )
                    t2 = time.perf_counter_ns()
                    duration_ns = t2 - t1
                    if tmp_rslt.returncode != 0:
                        error = "simulation failed with nonzero code"
                except subprocess.TimeoutExpired as e:
                    duration_ns = e.timeout / 1e9
                    error = "simulation timed out"
            if error is not None:
                # I have mixed feelings about directly printing the log
                # -> for an isolated error, in CI, it makes a lot of sense
                # -> but otherwise, it's just going to clutter up everything
                pytest.fail(
                    f"{error}:\n"
                    f"  command: {' '.join(args)}\n"
                    f"  exec_dir: {out_dir}\n"
                    f"  duration: {duration_ns * 1e9} seconds\n"
                    f"  code: {tmp_rslt.returncode}\n"
                    f"  output logfile: {log_file}"
                )
            # todo: give more thought to reusing simulation outputs:
            # - right now, we aren't reporting stdout (or the path to the file)
            # - (we probably should stop reporting the precise duration)
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


def _load_field(snap: os.PathLike, /, field: str | Sequence[str]):
    # ugh, this is inefficient, but is currently necessary to read simulations with
    # less than 3 dimensions

    dims = cholla_utils.get_native_root_attributes(snap)["dims"]
    ndims = int(np.sum(dims > 1))

    if ndims == 3:
        return cholla_utils.load_field(snap, field)

    # assume that all fields were written to a single output file
    if isinstance(field, str):
        field_seq = [field]
    else:
        field_seq = field

    field_dict = {}
    with h5py.File(snap, "r") as f:
        for field_name in field_seq:
            tmp = np.array(f[field_name][...])
            # convert from to native bit ordering
            if np.issubdtype(tmp.dtype, np.floating):
                dtype = f"f{np.finfo(tmp.dtype).bits // 8}"
            else:
                # we probably need to be more careful in this case
                dtype = f"i{np.iinfo(tmp.dtype).bits // 8}"
            field_dict[field_name] = tmp.astype(dtype)

    if field != field_seq:
        return field_dict[field]
    return field_dict


# Custom assertions
# -----------------


def _get_common_fields(
    target_path: os.PathLike,
    *,
    ref_path: os.PathLike | None = None,
    ref_fields: Collection[str] | None = None,
    error_if_diff: bool = True,
) -> list[str]:
    """Finds list of common field names"""
    if ref_path is None and ref_fields is None:
        raise ValueError("ref_path or ref_fields must be provided")
    elif (ref_path is not None) and (ref_fields is not None):
        raise ValueError("ref_path & ref_fields can't both be provided")
    elif ref_fields is None:
        ref_path = pathlib.Path(ref_path)
        if not ref_path.exists():
            pytest.fail(f"no file at: {ref_path}")
        ref_fields = cholla_utils.get_native_fields(ref_path)

    target_path = pathlib.Path(target_path)
    if not target_path.exists():
        pytest.fail(f"no file at: {target_path}")
    target_field_list = cholla_utils.get_native_fields(target_path)

    common, only_ref = [], []
    for field in ref_fields:
        if field in target_field_list:
            common.append(field)
        else:
            only_ref.append(field)

    if error_if_diff and (only_ref != [] or len(common) != len(target_field_list)):
        only_target = []
        for elem in target_field_list:
            if elem not in ref_fields:
                only_target.append(elem)
        if ref_path is None:
            ref_descr = "ref fields"
        else:
            ref_descr = f"fields in {ref_path}"
        pytest.fail(
            f"{ref_descr} (0) differ from fields in {target_path}\n"
            f" -> common fields: {common}\n"
            f" -> only in (0): {only_ref}\n"
            f" -> only in (1): {only_target}"
        )
    return common


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
    __tracebackhide__ = True  # <- # Hide traceback for pytest

    # validate that a snapshot exists at path and that it has expected keys
    fields = _get_common_fields(path, ref_fields=expected.keys(), error_if_diff=True)

    data = cholla_utils.load_field(path, fields)
    for field, val in expected.items():
        np.testing.assert_allclose(
            data[field],
            desired=val,
            err_msg=f"unexpected {field!r} value in {path}",
            atol=atol,
            rtol=rtol,
        )


def _build_err_msg(
    targets: tuple[os.PathLike, os.PathLike],
    err_msg: str,
    header: str = "Snapshots are not equal",
    names: tuple[str, str] = ("actual", "reference"),
) -> str:
    """Build the error message"""
    tmp = [f"\n{header}", err_msg]
    for target, name in zip(targets, names, strict=True):
        if isinstance(target, (bytes, bytearray)):
            path = target.decode("utf-8")
        else:
            path = str(target)
        tmp.append(f"{name}: {path}")
    return "\n".join(tmp)


def _pretty_err_vec_lines(
    Linf_errvec: Sequence[float], L1_errvec: Sequence[float], field_list: Sequence[str]
) -> Iterator[str]:
    # Generate nicely-formatted lines showing error vectors & corresponding field names
    # - there's a lot of value to doing this in order to make it easy to parse what
    #   went wrong from the test-failure
    # - after all, most interactions with a good, rigorous test is reading the
    #   associated error message

    # we're essentially, yielding lines to nicely show 3, side-by-side
    # The result will look something like:
    # >    ┏   ┓    ┏   ┓       ┏   ┓
    # > L∞=┃...┃ L1=┃...┃ names=┃...┃
    # >    ┃...┃    ┃...┃       ┃...┃
    # >    ┗   ┛    ┗   ┛       ┗   ┛

    # compute the max width for the stringified format of any component in each vector
    widths = {
        "w_Linf": max(len("{:g}".format(float(e))) for e in Linf_errvec),
        "w_L1": max(len("{:g}".format(float(e))) for e in L1_errvec),
        "w_name": max(len(e) for e in field_list),
    }

    # create kwargs conveying content & for boundary lines (i.e. first & last line)
    content_kw = dict(Linf_kind="g", L1_kind="g", braceL="┃", braceR="┃", **widths)
    bound_kw = dict(Linf="", Linf_kind="s", L1="", L1_kind="s", name="", **widths)

    blank_descr = dict(Linf_descr="   ", L1_descr="   ", name_descr="     ")
    with_descrs = dict(Linf_descr="L∞=", L1_descr="L1=", name_descr="name=")

    template = (
        "{Linf_descr:s}{braceL}{Linf:{w_Linf}{Linf_kind}}{braceR} "
        "{L1_descr:s}{braceL}{L1:{w_L1}{L1_kind}}{braceR} "
        "{name_descr:s}{braceL}{name:{w_name}s}{braceR}"
    )

    halfway = len(field_list) // 2
    yield template.format(braceL="┏", braceR="┓", **bound_kw, **blank_descr)
    for i, (Linf, L1, name) in enumerate(zip(Linf_errvec, L1_errvec, field_list)):
        descr = with_descrs if i == halfway else blank_descr
        yield template.format(Linf=Linf, L1=L1, name=name, **descr, **content_kw)
    yield template.format(braceL="┗", braceR="┛", **bound_kw, **blank_descr)


@dataclasses.dataclass
class ErrVecReqs:
    max_L1_errvec_mag: float = math.inf
    max_Linf_errvec_comp: float = math.inf

    def __post_init__(self):
        if self.max_L1_errvec_mag == math.inf and self.max_Linf_errvec_comp == math.inf:
            raise ValueError(
                "max_L1_errvec_mag & max_Linf_errvec_comp can't both be math.inf"
            )

    def _prep_err_header(self, hdr_prefix: str) -> str:
        # helper method to prepare the error message header
        _max_L1_str, _max_Linf_str = [
            "∞" if elem == math.inf else f"{float(elem):g}"
            for elem in (self.max_L1_errvec_mag, self.max_Linf_errvec_comp)
        ]
        return (
            f"{hdr_prefix} aren't equal such that:\n"
            f"-> L1-error vector's magnitude is under {_max_L1_str} AND\n"
            f"-> components of L∞-error vector are under {_max_Linf_str}"
        )


def _check_error_vec(
    actual_path: os.PathLike,
    ref_path: os.PathLike,
    hdr_prefix: str,
    err_vec_reqs: ErrVecReqs,
    names: tuple[str, str],
    err_msg: str = "",
):
    # TODO: it would be MUCH more useful for this function to construct and return an
    # ErrVecReport object and then subsequently check whether that object satisfies the
    # error requirements separately
    #
    # This provides 2 benefits:
    # 1. it reduces the size of this function
    # 2. it paves the way towards computing this information outside of an
    #    assertion-test. This would be useful for generating convergence plots (or
    #    setting up new tests). In this scenario, we would probably relocate the
    #    functionality for constructing the object to the cholla_utils package

    __tracebackhide__ = True  # <- # Hide traceback for pytest

    header = err_vec_reqs._prep_err_header(hdr_prefix=hdr_prefix)
    max_L1_errvec_mag = err_vec_reqs.max_L1_errvec_mag
    max_Linf_errvec_comp: float = err_vec_reqs.max_Linf_errvec_comp

    # it would be nice to slot this into the error-formatting
    field_list = _get_common_fields(actual_path, ref_path=ref_path, error_if_diff=True)

    try:
        field_list.remove("GasEnergy")
    except ValueError:
        pass

    if len(field_list) == 0:
        msg = _build_err_msg(
            (actual_path, ref_path),
            err_msg + "\nsnapshots don't have any comparable fields",
            header=header,
            names=names,
        )
        raise AssertionError(msg)

    # load in the fields
    ref_data = _load_field(ref_path, field_list)
    actual_data = _load_field(actual_path, field_list)

    # construct the Linf_errvec & L1_errvec
    # -------------------------------------
    Linf_errvec = []
    L1_errvec = []

    for field in field_list:
        ref = ref_data[field]
        actual = actual_data[field]
        if ref.shape != actual.shape:
            reason = (
                f'\nshapes {actual.shape}, {ref.shape} mismatch for "{field}" field'
            )
            msg = _build_err_msg(
                (actual_path, ref_path), err_msg + reason, header=header, names=names
            )
            raise AssertionError(msg)

        err = np.abs(ref - actual)
        # notes on summation logic:
        # -> the original C++ logic for this calculation used Kahan summation
        #    (which is slow, but has nice error properties)
        # -> we currently use np.sum, which uses pairwise summation at this time
        #    (less robust than Kahan summation, but better than naive summation)
        # -> if we're really worried about numerical accuracy we could:
        #    - look for a 3rd party module that computes a numpy sum with Kahan
        #      summation (this almost certainly exists as a c extension)
        #    - could use python's builtin `math.fsum` (this would be a LOT slower). I
        #      THINK the algorithm may changed in python 3.12, but my impression is
        #      that it's more accurate than Kahan summation
        L1_errvec.append(np.sum(err) / err.size)
        Linf_errvec.append(np.amax(err))

    # build up a list of lines detailing test failure
    # -> if the lines list remains empty, then there aren't any problems!
    lines = []

    def _check_err_vec_comps(errvec, max_val, errvec_name):
        prob_pairs = [
            (f, comp) for f, comp in zip(field_list, errvec) if not (comp < max_val)
        ]
        n = len(prob_pairs)
        if n != 0:
            lines.append(f"{n} component(s) of {errvec_name} aren't under {max_val!r}")
            lines.append(
                "-> associated field-names & values of violating components include:"
            )
            for pair in prob_pairs:
                lines.append(f"   {pair[0]}: {float(pair[1])!r}")

    _check_err_vec_comps(Linf_errvec, max_Linf_errvec_comp, "L∞-error vector")

    L1_errvec_mag = float(np.linalg.norm(L1_errvec))
    if not (L1_errvec_mag < max_L1_errvec_mag):
        lines.append(
            f"mag of L1-error vector, {L1_errvec_mag!r}, isn't under "
            f"{max_L1_errvec_mag}"
        )
        _check_err_vec_comps(L1_errvec, max_L1_errvec_mag, "L1-error vector")

    if len(lines) > 0:
        lines.append(
            "The L∞-error vector, L1-error vector, & corresponding fields names are:"
        )
        lines.extend(_pretty_err_vec_lines(Linf_errvec, L1_errvec, field_list))
        lines.append(f"magnitude of L1-error vector: {L1_errvec_mag!r}")
        msg = _build_err_msg(
            (actual_path, ref_path),
            err_msg + "\n" + "\n".join(lines),
            header=header,
            names=names,
        )
        raise AssertionError(msg)


def converge_error_test(
    actual_path: os.PathLike,
    ref_path: os.PathLike,
    *,
    err_msg: str = "",
    max_L1_errvec_mag: float = math.inf,
    max_Linf_errvec_comp: float = math.inf,
    expected_step_count: int | None,
    is_initial_final: bool = False,
):
    """
    Computes errors against a reference solution

    Let's provides some definitions
    - for a quantity ``q``, we can construct an array of errors where
      ``e[i,j,k] = |q_actual[i,j,k] - q_ref[i,j,k]|
    - let the L1-error of ``q`` specify the average magnitude from the array
      of errors (i.e. it's the L1-norm of the error array)
    - let the L∞-error of ``q`` specify the max magnitude from the array of
      errors (i.e. it's the L∞-norm of the error array)

    This field function constructs both an L1-error vector and a L∞-error vector,
    where each component specifies a distinct simulated field

    Parameters
    ----------
    actual_path, ref_path
        Path to the snapshots being compared
    max_L1_errvec_mag
        Specifies the max allowed value that the Euclidean norm of the L1 error vector
        is allowed to have. (the original C++ version of this logic called this
        ``maxAllowedL1Error``).
    max_Linf_errvec_comp
        Specifies the max allowed value that a component of the L∞ error vector can
        have (the original C++ version of this logic called this ``maxAllowedError``)
    """
    __tracebackhide__ = True  # <- # Hide traceback for pytest

    if is_initial_final:
        names = ("final snap", "initial snap")
        hdr_prefix = "initial & final snapshots"
    else:
        names = ("actual snap", "ref snap")
        hdr_prefix = "actual & ref snapshots"

    err_vec_reqs = ErrVecReqs(
        max_L1_errvec_mag=max_L1_errvec_mag, max_Linf_errvec_comp=max_Linf_errvec_comp
    )

    # I'm not sure we should be making this check
    # -> it feels like we're over-constraining the test
    # -> does it really matter if the number of steps slightly changes if the answers
    #    remain robust?
    if expected_step_count is not None:
        actual_count = cholla_utils.get_native_root_attributes(actual_path)["n_step"][0]
        if actual_count != expected_step_count:
            header = err_vec_reqs._prep_err_header(hdr_prefix=hdr_prefix)
            reason = (
                f"\n{names[0]} written after {actual_count} steps "
                f"(expected: {expected_step_count} steps)"
            )
            msg = _build_err_msg(
                (actual_path, ref_path), err_msg + reason, header=header, names=names
            )
            raise AssertionError(msg)

    _check_error_vec(
        actual_path=actual_path,
        ref_path=ref_path,
        hdr_prefix=hdr_prefix,
        err_vec_reqs=err_vec_reqs,
        names=names,
        err_msg=err_msg,
    )


def _ulps_dist(a, b):
    if a.dtype == np.float64 or b.dtype == np.float64:
        if a.dtype != np.float64:
            a = a.astype(np.float64)
        elif b.dtype != np.float64:
            b = b.astype(np.float64)
        itype = np.int64
    elif a.dtype == np.float32 or b.dtype == np.float32:
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        elif b.dtype != np.float32:
            b = b.astype(np.float32)
        itype = np.int32
    else:
        raise ValueError("neigther a nor b is a float64")

    w_finite = np.logical_and(np.isfinite(a), np.isfinite(b))
    # just like the original c++ code, we don't worry about changing signs,
    # but we could handle that properly in the future
    w_samesign = (a < 0) == (b < 0)
    w_cmp = np.logical_and(w_finite, w_samesign)

    bitcast_a = a.view(itype)
    bitcast_b = b.view(itype)

    out = np.empty(shape=a.shape, dtype=itype)
    out[w_cmp] = np.abs(bitcast_a[w_cmp] - bitcast_b[w_cmp])
    out[~w_cmp] = np.iinfo(itype).max
    return out


@dataclasses.dataclass
class ClassicAbsErrReqs:
    # A pair of elements are considered if either:
    # - the magnitude of their difference is less than `atol` OR
    # - the elements differ by no more than `maxulp` number of units in the last place
    #
    # NOTE: this only exists for historical purposes. I think it makes more sense to
    #       transition to using `atol` and `rtol`

    # in the original C++, there was a comment that basically said:
    # > [atol] is changed from the default [(1.e-14)] since AMD/Clang
    # > appear to differ from NVIDIA/GCC/XL by roughly 1e-12
    atol: float = 5.0e-12

    maxulp: int = 4

    def _prep_err_header(self, hdr_prefix: str) -> str:
        # helper method to prepare the error message header
        return (
            f"{hdr_prefix} aren't equal such that each element:\n"
            f"-> has an absolute difference under {self.atol} OR\n"
            f"-> differ by no more than {self.maxulp} units in the last place"
        )


def _check_classic_abs_err(
    actual_path: os.PathLike,
    ref_path: os.PathLike,
    hdr_prefix: str,
    err_reqs: ClassicAbsErrReqs,
    names: tuple[str, str],
    err_msg: str = "",
):
    __tracebackhide__ = True  # <- # Hide traceback for pytest

    # this doesn't work yet with particles or gravitational potential
    header = err_reqs._prep_err_header(hdr_prefix)

    # it would be nice to slot this into the error-formatting
    field_list = _get_common_fields(actual_path, ref_path=ref_path, error_if_diff=True)

    if len(field_list) == 0:
        msg = _build_err_msg(
            (actual_path, ref_path),
            err_msg + "\nsnapshots don't have any comparable fields",
            header=header,
            names=names,
        )
        raise AssertionError(msg)

    # load in the fields
    ref_data = _load_field(ref_path, field_list)
    actual_data = _load_field(actual_path, field_list)

    # todo: consider building a report for all fields rather than just reporting the
    #       issue with the first field that we see
    for field in field_list:
        ref = ref_data[field]
        actual = actual_data[field]
        if ref.shape != actual.shape:
            reason = (
                f'\nshapes {actual.shape}, {ref.shape} mismatch for "{field}" field'
            )
            msg = _build_err_msg(
                (actual_path, ref_path), err_msg + reason, header=header, names=names
            )
            raise AssertionError(msg)

        err = np.abs(ref - actual)
        nulps = _ulps_dist(ref, actual)

        are_unequal = ~np.logical_or(err < err_reqs.atol, nulps <= err_reqs.maxulp)
        num_unequal_elem = int(are_unequal.sum())

        if num_unequal_elem != 0:
            issues = [
                f"problematic field: {field}\nmax abs diff: {float(err.max())}\n",
                f"max ULP diff: {nulps.max()}\nviolation count: {num_unequal_elem}\n",
            ]

            indices = np.argwhere(are_unequal)

            _max_shown = 3
            if num_unequal_elem == 1:
                issues.append(f"Violation at index: {indices[0, :]}\n")
                issues.append("Values at index:\n")
            else:
                if num_unequal_elem <= _max_shown:
                    issues.append("Violation at indices:\n")
                else:
                    issues.append(f"First {_max_shown} violations at indices:\n")
                pretty_idx_list = [tuple(idx) for idx in indices[:_max_shown].tolist()]
                issues.append(f"{pretty_idx_list}\n")
                issues.append("Values at indices:\n")

            for arr, name in zip([actual, ref], names, strict=True):
                issues.append(f"  {name}: {arr[are_unequal][:_max_shown]}\n")

            msg = _build_err_msg(
                (actual_path, ref_path),
                err_msg + "\n" + "".join(issues),
                header=header,
                names=names,
            )
            raise AssertionError(msg)


def perform_answer_test(
    actual_path: os.PathLike,
    ref_path: os.PathLike,
    expected_step_count: int | None,
    err_spec: ErrVecReqs | ClassicAbsErrReqs,
):
    __tracebackhide__ = True  # <- # Hide traceback for pytest
    names = ("test-sim snap", "fiducial snap")
    hdr_prefix = "test-sim & fiducial snapshots"

    if expected_step_count is not None:
        actual_count = cholla_utils.get_native_root_attributes(actual_path)["n_step"][0]
        if actual_count != expected_step_count:
            header = err_spec._prep_err_header(hdr_prefix)
            reason = (
                f"\n{names[0]} written after {actual_count} steps "
                f"(expected: {expected_step_count} steps)"
            )
            msg = _build_err_msg(
                (actual_path, ref_path), "\n" + reason, header=header, names=names
            )
            raise AssertionError(msg)

    if isinstance(err_spec, ErrVecReqs):
        _check_error_vec(
            actual_path=actual_path,
            ref_path=ref_path,
            hdr_prefix=hdr_prefix,
            err_vec_reqs=err_spec,
            names=names,
        )
    elif isinstance(err_spec, ClassicAbsErrReqs):
        _check_classic_abs_err(
            actual_path=actual_path,
            ref_path=ref_path,
            hdr_prefix=hdr_prefix,
            err_reqs=err_spec,
            names=names,
        )
