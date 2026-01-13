"""
Logic for constructing a cache of modification times.
"""

from collections.abc import Iterator
import dataclasses
import json
import os

from .run_doxygen import DoxyBuildPaths


@dataclasses.dataclass
class DoxyBuildSnapshot:
    """
    Represents a snapshot of a doxygen build
    """

    # lists paths to directories or individual files that are relevant to
    # the build
    build_paths: DoxyBuildPaths

    # holds the mtimes for each entry in build_paths
    mtimes: dict[str, int]

    # list of files constructed within dox_build_dir (to be clear, they should
    # only contain the basename of each file)
    build_dir_artifacts: set[str]

    def __post_init__(self):
        assert self.build_dir_artifacts is not None  # sanity check!

    def write_json(self, path):
        d = {
            "build_paths": self.build_paths.to_serialization_dict(),
            "mtimes": self.mtimes,
            "build_dir_artifacts": list(self.build_dir_artifacts),
        }
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            build_paths=DoxyBuildPaths(**data["build_paths"]),
            mtimes=data["mtimes"],
            build_dir_artifacts=set(data["build_dir_artifacts"]),
        )


def _it_tree_names(dir_path: os.PathLike) -> Iterator[str]:
    """
    Recursively walk the names of file or directory in the specified path

    The name is the location relative to dir_path. To get the path from the
    name, use `os.path.join(dir_path, name)`
    """
    dir_path = os.path.normpath(str(dir_path))
    n_dir_path_chars = len(dir_path)
    for root_path, dirs, files in os.walk(dir_path, followlinks=False):
        # root_path holds the path to root
        # root_name holds the path to root relative to dir_path
        if root_path == dir_path:
            root_name = ""
        else:
            assert root_path[n_dir_path_chars] == os.sep  # sanity_check
            root_name = root_path[n_dir_path_chars + 1 :]

        yield from (os.path.join(root_name, d) for d in dirs)
        yield from (os.path.join(root_name, f) for f in files)


def _get_mtime(nominal_path: os.PathLike, return_nameset: bool = False):
    """
    Walk a directory and determine the most recent time at which a contained
    file/directory was modified, created, deleted, removed, etc.

    Parameters
    ----------
    nominal_path
        The nominal path to measure the mtime for
    return_nameset
        When True, returns set holding the locations of all queried
        files/directory, relative to nominal_path, other than
        nominal_path itself.
    """

    def _fn(path):  # gives posix timestamp in seconds (rounded up)
        return int(os.stat(path).st_mtime + 1)

    # explicitly measure mtime of nominal_path
    max_mtime = _fn(nominal_path)
    nameset = set()

    if os.path.isdir(nominal_path):
        # make iterator over the mtimes of each item in dir_path. We explicitly
        # check mtimes of directories since they provide the only indication that
        # files within a that directory were deleted/moved
        for name in _it_tree_names(nominal_path):
            if return_nameset:
                nameset.add(name)
            max_mtime = max(_fn(os.path.join(nominal_path, name)), max_mtime)
    if return_nameset:
        return max_mtime, nameset
    return max_mtime


def try_measure_snap(
    build_paths: DoxyBuildPaths, loudly_fail: bool = False
) -> None | DoxyBuildSnapshot:
    """
    try to measure the mtime (modification time) for each the specified
    dependenies/artifacts of the doxygen process

    Parameters
    ----------
    build_paths
        Holds the paths for each of the dependencies and artifacts of the
        doxygen build
    loudly_fail
        When True, this will raise an exception upon failure

    Returns
    -------
    None or DoxyBuildPaths[int]
        If one or more of the provided paths don't exist, then None is
        returned. Otherwise, a DoxyBuildPaths, where each field holds
        the appropriate modification time, is returned
    """
    mtimes = {}
    build_dir_artifacts = None

    for field in dataclasses.fields(build_paths):
        name = field.name
        path = getattr(build_paths, name)

        if not os.path.exists(path):
            return None
        elif name == "dox_build_dir":
            mtimes[name], build_dir_artifacts = _get_mtime(path, True)
        mtimes[name] = _get_mtime(path, False)

    return DoxyBuildSnapshot(
        build_paths=build_paths, mtimes=mtimes, build_dir_artifacts=build_dir_artifacts
    )


def build_consistent_with_cache(
    cache_file: os.PathLike, depend_artifact_paths: DoxyBuildPaths
) -> bool:
    """
    Checks
    Checks whether the modification times for each dependency or artifact of
    the doxygen build matches the modification times from a previous build

    Parameters
    ----------
    cache_file
        Path to file where a DoxyBuildSnapshot would have been saved by a
        previous snapshot
    depend_artifact_paths
        Holds the paths for each of the dependencies & artifacts of the
        doxygen build

    Returns
    -------
    bool
        Indicates whether there was a match
    """

    # load the cached_mtimes
    try:
        cached_snap = DoxyBuildSnapshot.from_json(cache_file)
    except (FileNotFoundError, KeyError):
        return False  # if we can't find the cached mtimes, we need to call doxygen

    # measure the modification times
    actual_snap = try_measure_snap(depend_artifact_paths)

    if actual_snap is None:
        # one or more dependencies/artifacts is missing.
        return False

    return actual_snap == cached_snap
