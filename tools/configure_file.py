#!/usr/bin/env python3
"""
A simple tool for configuring template files. It is modelled after the functions with
the same name in CMake and Meson.

Detailed documentation is maintained at
https://cholla.readthedocs.io/en/latest/Development/configure_file.html.
"""

# If you modify the behavior of this tool, PLEASE make sure that change is reflected in
# the source file.
#
# this script was originally written to be used with Grackle (it has gone through a fair
# amount of evolution since then)

import argparse
import os
import re
import sys
from typing import (
    Container, Dict, Iterator, List, Optional, Sequence, Set, Tuple
)

if sys.version_info < (3, 6):
    # for added context, we use fstrings, which were introduced in python 3.6
    raise ValueError("python 3.6 or newer is required to run this script")


_MAX_VARNAME_SIZE = 256
_VALID_VARNAME_STR = "\\w{{1,{}}}".format(_MAX_VARNAME_SIZE)
_PATTERN = re.compile(r"(@{}@)|(@[^\s@]*@?)".format(_VALID_VARNAME_STR))
_VALID_VARNAME_DESCR = (
    f"A valid varname is composed of 1 to {_MAX_VARNAME_SIZE} alphanumeric ASCII "
    "characters. An alphanumeric character is an uppercase or lowercase letter (A-Z "
    "or a-z), a digit (0-9) or an underscore (_)"
)


def is_valid_varname(
    s: str, start: Optional[int] = None, stop: Optional[str] = None
) -> bool:
    """Returns whether ``s[slice(start,stop)] is a valid variable name
    """
    return re.fullmatch(_VALID_VARNAME_STR, s[slice(start, stop)]) is not None

class Substituter:
    """Performs variable substitution"""

    # holds the the mapping of variable names to values
    # -> this object is never mutated
    # -> reminder: a value of None
    _variable_map: Dict[str, Optional[str]]
    # this set is used to track the names of all variables accessed from _variable_map
    used_variable_set: Set[str]

    def __init__(self, variable_map: Dict[str, Optional[str]]):
        self._variable_map = variable_map
        self.used_variable_set = set()
        self._variable_map = variable_map

    def _replace_at_signs(self, line: str, line_num:int) -> Tuple[bool, str]:
        """
        Returns a version of ``line`` after performing all (if any) @-sign substitutions
        """

        match_count = 0
        prev_pos = 0
        chunks = []
        for matchobj in _PATTERN.finditer(line):
            # append text between prev_pos and match.start
            chunks.append(line[prev_pos:matchobj.start()])

            # append the replacement
            if matchobj.lastindex != 1:
                err_msg = (
                    f"{matchobj[0]!r}, the string starting with occurence number "
                    f"{2 * match_count + 1} of the '@' character on line number "
                    f"{line_num} doesn't specify a valid variable name. "
                    f"{_VALID_VARNAME_DESCR}"
                )
                return False, err_msg
            varname = matchobj[1][1:-1]
            try:
                value = self._variable_map[varname]
            except KeyError:
                err_msg = (
                    f"the variable {varname} (specified by a string enclosed by a "
                    f"pair of '@' characters on line {line_num}) wasn't provided by "
                    "the caller"
                )
                return False, err_msg

            if value is None:
                err_msg = (
                    f"the variable {varname} (specified by a string enclosed by a "
                    f"pair of '@' characters on line {line_num}) was defined by the "
                    f"caller without specifying a value"
                )
                return False, err_msg

            chunks.append(value)
            self.used_variable_set.add(varname)

            # update prev_pos
            prev_pos = matchobj.end()

        chunks.append(line[prev_pos:])
        return True, "".join(chunks)

    def __call__(self, line: str, line_num: int) -> Tuple[bool, str]:
        """Performs the substitution on ``line``

        The trailing newline should have been stripped off of line before
        calling this method

        Returns a pair of value. If the first value is a boolean denoting
        whether the operation succeeded. Upon success, the second value is
        the substituted line. Upon failure, the second value is an
        error message
        """
        m = re.match(r"^[ \t]*#[ \t]*configurefile_define[ \t]*", line)
        if (m is not None) and m.group(0) == line:
            err_msg = (
                f"line {line_num} has #configurefile_define but doesn't specify a "
                "macro name"
            )
            return False, err_msg
        elif (m is not None) and m.group(0)[-1].isspace():
            tmp = line[m.end():].rstrip().split()
            if len(tmp) != 1:  # we can't have `#configurefile_define <var> <more...>
                return False, f"line {line_num}, `{line}`, has an invalid format"
            varname = tmp[0]
            try:
                value = self._variable_map[varname]
            except KeyError:
                return True, f"/* #undef {varname} */"
            self.used_variable_set.add(varname)
            if value is None:
                return True, f"#define {varname}"
            return True, f"#define {varname} {value}"
        else:
            return self._replace_at_signs(line, line_num)

def configure_file(
    lines: Iterator[str],
    variable_map: Dict[str, Optional[str]],
    out_fname: str,
    literal_linenos: Container[str]
):
    """
    Writes a new file to out_fname, line-by-line, while performing variable
    substituions
    """

    subber = Substituter(variable_map=variable_map)
    out_f = open(out_fname, "w")

    for line_num, line in enumerate(lines):
        # make sure to drop any trailing '\n'
        assert line[-1] == "\n", "sanity check!"
        line = line[:-1]
        if line_num in literal_linenos:
            subbed = line
        else:
            success, tmp = subber(line, line_num)
            if not success:
                out_f.close()
                os.remove(out_fname)
                raise RuntimeError(tmp)
            subbed = tmp
        out_f.write(subbed)
        out_f.write("\n")

    unused_variables = subber.used_variable_set.symmetric_difference(variable_map)

    if len(unused_variables) > 0:
        os.remove(out_fname)
        raise RuntimeError(
            f"the following variable(s) were provided, but unused: {unused_variables!r}"
        )


def _parse_variables(
    dict_to_update: Dict[str, Optional[str]],
    var_val_assignment_str_l: List[str],
    val_is_file_path: bool=False
):
    for var_val_assignment_str in var_val_assignment_str_l:
        stripped_str = var_val_assignment_str.strip()  # for safety

        # so the the contents should look like "<VAR>=<VAL>"

        n_equal = stripped_str.count("=")
        if n_equal == 0:
            if val_is_file_path:
                raise RuntimeError(
                    f"{stripped_str!r} is an invalid argument for associated a "
                    "variable with the contents of a file"
                )
            var_name = stripped_str
            value = None
        elif (n_equal != 1) and val_is_file_path:
            raise RuntimeError(
                f"{stripped_str} doesn't specify a variable-value pair: it contains "
                "multiple '=' characters"
            )
        else:
            var_name, value = stripped_str.split("=", maxsplit=1)

        if not is_valid_varname(var_name):
            raise RuntimeError("{!r} is not a valid variable name".format(var_name))
        elif var_name in dict_to_update:
            raise RuntimeError(
                "the {!r} variable is defined more than once".format(var_name)
            )

        if val_is_file_path:
            path = value
            if not os.path.isfile(path):
                raise RuntimeError(
                    "error while trying to associate the contents of the file at"
                    "at {path!r} with the {var_name!r} variable: no such file exists"
                )
            with open(value, "r") as f:
                # we generally treat the characters in the file as literals
                # -> we do need to make a point of properly escaping the
                #    newline characters
                assert os.linesep == "\n"  # implicit assumption
                value = f.read().replace(os.linesep, r"\n")
        dict_to_update[var_name] = value


def main(arg_sequence: Optional[Sequence[str]] = None):
    """
    Drives the program

    Parameters
    ----------
    arg_sequence: sequence of str, optional
        List of arguments to parse. When this is None, arguments are parsed
        from sys.argv
    """
    args = parser.parse_args(args=arg_sequence)
    # handle clobber-related logic
    clobber, out_fname = args.clobber, args.output
    if os.path.isfile(out_fname) and not clobber:
        raise RuntimeError(
            f"A file already exists at {out_fname!r}. Use --clobber, to overwrite"
        )

    # fill variable_map with the specified variables and values
    variable_map: Dict[str, Optional[str]] = {}
    _parse_variables(variable_map, args.variables, val_is_file_path=False)
    _parse_variables(
        variable_map, args.variable_use_file_contents, val_is_file_path=True
    )

    literal_linenos = set()
    if args.literal_linenos is not None:
        literal_linenos = set(args.literal_linenos)
    # use variable_map to actually create the output file
    with open(args.input, "r") as f_input:
        line_iterator = iter(f_input)
        configure_file(
            lines=line_iterator,
            variable_map=variable_map,
            out_fname=out_fname,
            literal_linenos=literal_linenos,
        )

    return 0


parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument(
    "--variable-use-file-contents",
    action="append",
    default=[],
    metavar="VAR=path/to/file",
    help=(
        "associates the (possibly multi-line) contents contained by the "
        "specified file with VAR"
    ),
)

parser.add_argument(
    "-D",
    action="append",
    default=[],
    dest="variables",
    metavar="VAR=VAL",
    help="associates the value, VAL, with the specified variable, VAR",
)
parser.add_argument("-i", "--input", required=True, help="path to input template file")
parser.add_argument("-o", "--output", required=True, help="path to output file")
parser.add_argument(
    "--clobber",
    action="store_true",
    help="overwrite the output file if it already exists",
)
parser.add_argument(
    "--literal-linenos",
    nargs="*",
    type=int,
    help=(
        "line numbers corresponding to lines that are treated as literals "
        "(i.e. no variable substitution is performed)"
    ),
)

if __name__ == "__main__":
    sys.exit(main())
