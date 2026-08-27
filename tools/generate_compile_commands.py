#!/usr/bin/env python3
"""
A simple tool for configuring a compile_commands.json file.

This file-format is understood by clang-tidy
"""

import argparse
import dataclasses
import itertools
import json
import os
import re
import sys
from typing import Iterable, Iterator, List, Optional, Tuple


def _skip_nvcc_compile_compile_args(compile_args: Iterable[str]) -> Iterator[str]:
    """Filters out any nvcc-specific compile commands."""
    # -> we may need to add more known options over time
    # -> this is written in a generic enough way that we could factor out some logic
    #    if we wanted to do a similar kind of thing for HIP compiler args
    boolean_opts = ["--expt-extended-lambda"]
    opts_with_1_arg = ["-ccbin", "-fmad"]

    # build up regex-matchers
    _self_contained_opts = [re.escape(opt) for opt in boolean_opts]
    _argpair_leaders = [re.escape(opt) for opt in opts_with_1_arg]
    if _argpair_leaders:
        _self_contained_opts.append(f"({'|'.join(_argpair_leaders)})=.+")
    _argpair_leader_matcher = re.compile(f"^({'|'.join(_argpair_leaders)})$")
    _self_contained_matcher = re.compile(f"^({'|'.join(_self_contained_opts)})$")

    itr = iter(compile_args)
    for elem in itr:
        if _self_contained_matcher.match(elem):
            # print(f"skipping: {elem!r}")
            continue
        elif _argpair_leader_matcher.match(elem):
            # the next line consumes the 2nd element of the argument-pair
            # (we suppress the ruff-lint about the variable being unused)
            second = next(itr)  # noqa: F841
            # print(f"skipping: {elem!r} {second!r}")
            continue
        else:
            yield elem


@dataclasses.dataclass
class DatabaseEntry:
    directory: str
    file: str
    arguments: str
    output: Optional[str]

    def unique_key(self) -> Tuple[str, Optional[str]]:
        def _coerce(path):
            if path is None:
                return None
            elif os.path.isabs(path):
                return os.path.abspath(path)
            else:
                return os.path.abspath(os.path.join(self.directory, path))

        return (_coerce(self.file), _coerce(self.output))


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DatabaseEntry):
            itr = dataclasses.asdict(obj).items()
            out = {k: v for k, v in itr if v is not None}
            return out
        # Let the base class default method raise the TypeError
        return super().default(obj)


def prepare_database_entries(args: argparse.Namespace) -> Iterator[DatabaseEntry]:
    compiler_opts = args.compiler_opts
    if args.strip_nvcc_flags:
        compiler_opts = list(_skip_nvcc_compile_compile_args(compiler_opts))
    leading_args = [args.compiler] + compiler_opts

    for source in args.sources:
        tmp = os.path.splitext(source)
        assert tmp[1] != ""
        output = f"{tmp[0]}{args.outputs_suffix}"
        yield DatabaseEntry(
            directory=args.directory,
            file=source,
            arguments=leading_args + ["-c", source, "-o", output],
            output=output,
        )


def main(arg_list: Optional[List[str]] = None):
    arg_list = sys.argv[1:] if arg_list is None else arg_list

    parser = argparse.ArgumentParser(
        description="prepare compile_commands.json file",
        usage="%(prog)s [flags] -- [COMPILER_OPTS]",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="The path of the generated compile commands database",
    )
    parser.add_argument(
        "--directory", required=True, help="Working directory of the compilation."
    )
    parser.add_argument(
        "--compiler",
        required=True,
        help=(
            "Path to the compiler used with the command. If this is a relative path, "
            "it should be relative to the working directory"
        ),
    )
    parser.add_argument(
        "--sources", required=True, nargs="+", help="list of source files"
    )
    parser.add_argument(
        "--outputs-suffix",
        required=True,
        help="The suffix of the output file produced by a compilation command",
    )
    parser.add_argument(
        "--prepend-entries-from",
        nargs="+",
        required=False,
        help=(
            "Paths to files that compilation databases are read from. This option "
            "only exists for the purpose of concatenation"
        ),
    )
    parser.add_argument(
        "--strip-nvcc-flags", action="store_true", help="remove known nvcc flags"
    )
    parser.add_argument(
        "compiler_opts",
        nargs="*",
        help=(
            "The options passed to the compiler in each compilation command. This "
            "should not include the path to the input file or specify the output "
            "file."
        ),
    )

    # we manually intervene to make sure the caller passes -- to delimit between
    # options to this tool and forwarded args (I don't want to play games)
    if "--help" in arg_list:
        parser.parse_args(["--help"])
    else:
        try:
            hyphen_sentinel = arg_list.index("--", 1)
        except ValueError:
            hyphen_sentinel = None
            parser.error("arg list doesn't feature compiler-options that follow a `--`")
        args = parser.parse_args(arg_list[:hyphen_sentinel] + ["--", "dummy"])
        args.compiler_opts = arg_list[hyphen_sentinel + 1 :]

    entry_source_itr = itertools.chain(
        [] if args.prepend_entries_from is None else args.prepend_entries_from, [None]
    )

    entry_origins = {}
    database = []
    for path in entry_source_itr:
        if path is not None:
            origin_descr = f"loaded from {path}"
            with open(path, "r") as f:
                new_entries = json.load(f, object_hook=lambda d: DatabaseEntry(**d))
        else:
            origin_descr = "specified via cli"
            new_entries = prepare_database_entries(args)
        for entry in new_entries:
            key = entry.unique_key()
            if key in entry_origins:
                raise RuntimeError(
                    f"the {key}-entry {origin_descr} conflicts with the entry "
                    f"{entry_origins[key]}"
                )
            entry_origins[key] = origin_descr
            database.append(entry)
    with open(args.output_file, "w") as f:
        json.dump(database, f, indent=2, cls=MyEncoder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
