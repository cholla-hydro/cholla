#!/usr/bin/env python3
"""
Convert a legacy parameter file so the string-parameters are now quoted.
Importantly, the program assumes that the file format is valid.
"""

# for portability, only use built-in package packages present in python 3.7
import argparse
from contextlib import nullcontext
import shutil
import sys
from tempfile import NamedTemporaryFile
from typing import IO, Iterator, NamedTuple, Optional


class LineInfo(NamedTuple):
    """This holds the contents for a line in the file"""

    lineno: int
    # this is the full line (including the trailing newline)
    line: str
    # if line defines a parameter, param_name holds the fully qualified name
    param_name: Optional[str] = None
    # if line defines a parameter, this holds the rhs of the equal sign
    rhs: Optional[str] = None


class DecodeError(ValueError):
    # loosely inspired by the standard library's JSONDecodeError & TOMLDecodeError
    def __init__(self, msg: str, doc: str, lineno: int):
        ValueError.__init__(self, f"{msg}: line {lineno}")
        self.msg = msg
        self.doc = doc
        self.lineno = lineno


def read_lines(f: IO[str]) -> Iterator[LineInfo]:
    cur_table_header = None

    src = f.read()
    for lineno, line in enumerate(src.splitlines(keepends=True), start=1):
        if line.isspace() or line.startswith("#"):  # <- empty line or a comment
            yield LineInfo(lineno, line=line, param_name=None)

        elif line.startswith("["):
            _stripped_line = line.rstrip()
            if _stripped_line[-1] != "]":
                raise DecodeError("'[' must be closed by ']' before \\n", src, lineno)
            elif _stripped_line == "[]":
                raise DecodeError("empty table-names aren't allowed", src, lineno)

            _table_name = _stripped_line[1:-1]
            _dot_prob = (
                _table_name[0] == "." or _table_name[-1] == "." or ".." in _table_name
            )
            if _dot_prob or "[" in _table_name or "]" in _table_name:
                raise DecodeError(f"invalid table name: {_table_name}", src, lineno)

            # for our purposes, it's ok to simply assume that the table-name doesn't
            # appear more than once and that neither the current table-name nor any
            # parent collides with any parameter-names
            cur_table_header = _table_name
            yield LineInfo(lineno, line=line, param_name=None)

        else:  # we assume that we have a key-value pair
            _param_name, equal, rhs = line.partition("=")

            if equal == "":  # <- there is no equal sign
                raise DecodeError(f"invalid line: {_table_name}", src, lineno)
            elif "." in _param_name or "" == _param_name:
                _msg = "parameter names in parameter files can't contain a '.'"
                raise DecodeError(_msg, src, lineno)
            elif "" == _param_name:
                raise DecodeError(
                    "'=' can't be the first character on a line", src, lineno
                )
            if cur_table_header is None:
                param_name = _param_name
            else:
                param_name = ".".join([cur_table_header, param_name])

            # for our purposes, it's ok to simply assume that the parameter name
            # doesn't appear more than once and doesn't collide with any table names
            yield LineInfo(lineno, line=line, param_name=param_name, rhs=rhs.rstrip())


_STRING_PARAMS = {
    "chemistry.kind",
    "chemistry.data_file",
    "init",
    "custom_bcnd",
    "outdir",
    "snr_filename",
    "sw_filename",
    "scale_outputs_file",
    "UVB_rates_file",
    "analysis_scale_outputs_file",
    "analysisdir",
    "skewersdir",
    # technically, these were not part of the dev-branch before the transition,
    # but, they existed in a parallel branch, that was used to run multiple
    # simulations (they are introduced in PR #386)
    "feedback.boundary_strategy",
    "feedback.snr_filename",
    "feedback.sn_model",
    "feedback.sn_rate",
}


def try_extract_quote_contents(s: str):
    if s[:1] != '"':
        return None
    _search_pos = 1
    while _search_pos < len(s):
        close_pos = s.find('"', _search_pos)
        if close_pos == -1:
            break
        elif s[close_pos - 1] != "\\":
            return s[1:close_pos]
        _search_pos = close_pos + 1
    return None


def try_extract_quoted_val_from_line(rhs: str):
    tmp = try_extract_quote_contents(rhs.rstrip())
    if (tmp is None) or (len(rhs) != (len(tmp) + 2)):
        return None
    return tmp


_CHARS_TO_BACKSLASH_ESCAPE = {
    "\b": "b",
    "\t": "t",
    "\n": "n",
    "\f": "f",
    "\r": "r",
    "\x1b": "e",  # <- '\e' is not recognized by python
    '"': '"',
    "\\": "\\",
}


def make_quoted_string(s):
    # this is insanely inefficient
    parts = ['"']
    for char in s:
        tmp = _CHARS_TO_BACKSLASH_ESCAPE.get(char)
        if tmp is not None:
            parts.append(f"\\{tmp}")
        else:
            parts.append(char)
    parts.append('"')
    return "".join(parts)


def _drive_work(f_in: IO[str], f_out: IO[str], skip_full_precheck: bool = False):
    # first let's do a quick and check to see if the lines are already quoted
    # (with the benefit of hindsight, I'm not so sure that this is worth doing)
    for info in read_lines(f_in):
        if info.param_name is not None:
            if "\\" in info.rhs:
                # to my knowledge this never happens. If it does, we can manually
                # update the file ourself (it's probably an indication that something
                # weird is going on
                raise RuntimeError(f"line {info.lineno}, `{info.line}` has a backslash")
            if not skip_full_precheck:
                tmp = try_extract_quoted_val_from_line(rhs=info.rhs)
                if tmp is not None:
                    raise RuntimeError(
                        f"line {info.lineno}, `{info.line}` already seems to have a "
                        "quoted value. Because its ambiguous whether the value should "
                        "actually include the quote, we cannot proceed"
                    )

    f_in.seek(0)
    for info in read_lines(f_in):
        if (info.param_name is None) or (info.param_name not in _STRING_PARAMS):
            f_out.write(info.line)
            continue
        quoted_value = make_quoted_string(info.rhs.rstrip())
        _equal_pos = info.line.find("=")
        assert _equal_pos != -1
        f_out.write(f"{info.line[:_equal_pos]}={quoted_value}\n")


def main(args: argparse.Namespace) -> int:
    with open(args.path, "r") as f_in:
        cm = NamedTemporaryFile() if (args.output != "-") else nullcontext(sys.stdout)
        with cm as f_out:
            _drive_work(f_in, f_out, args.skip_full_precheck)
            f_out.flush()
            if f_out is not sys.stdout:
                shutil.copy(src=f_out.name, dst=args.output)
    return 0


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", help="path to file that will be read")
parser.add_argument(
    "-o",
    "--output",
    default="-",
    help="specifies path to output file (writes to stdout by default)",
)
parser.add_argument(
    "--skip-full-precheck",
    action="store_true",
    help="skip the full check for whether the file contains quoted values",
)

if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
