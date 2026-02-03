# Developing `cholla_utils`

## Development Dependencies

There are a handful of additional packages required purely for developing `cholla_utils`.
These are dependencies for building docs and for running tests.

If you are managing your python environment with an up-to-date version of `pip` (25.1 or newer), you install these dependencies by invoking a command of the following form
:::{code-block} sh
python -m pip install --user --group <dependency-group>
:::
from the root of the repository (i.e. the directory containing {repository-file}`pyproject.toml`).
You would need to replace `<dependency-group>` with the name of a "dependency-group".
For example, replacing it with
- `test` will install all packages required for `cholla_utils`'s testing requirements
- `docs` will install all packages for building the documentation
- `dev` will install all development dependencies

The above command will install the dependencies independently of `cholla_utils` (i.e. the command doesn't care whether `cholla_utils` is installed and won't try to install it).
To install these dependencies at the same time as you build `cholla_utils`, you can instead use a command of the form
:::{code-block} sh
python -m pip install --user --group <dependency-group> -e .
:::
The `-e` flag (which enables an editable install) is entirely optional.


## Testing

The `cholla_utils` utility package has been developed with its own test-suite that is automatically run in Continuous Integration.
This test suite has been implemented using the standard [`pytest`](https://docs.pytest.org/en/stable/index.html) testing library.

### Running the Tests

To run the `cholla_utils` test suite (after you installed `cholla_utils` and testing dependencies), you can simply invoke the following command:

:::{code-block}sh
pytest
:::

from the root of the directory (i.e. the directory containing {repository-file}`pyproject.toml`).
If you encounter issues, you could try replacing `pytest` with `python -m pytest`.

### Helpful Tips

Pytest's command line interface is very powerful that lets you do a bunch of useful things.
For example, to run a particular test from a given module, you could invoke:

:::{code-block}sh
pytest path/to/test_mod.py::test_function
:::

If you want to test a variant of a parameterized test, you could invoke something like

:::{code-block}sh
pytest path/to/test_mod.py::test_function[paramA,paramB]
:::

:::{important}
If you are using zsh (common if you're using macOS), you may need to enclose the name of the test in single quotes
:::

There are also 2 noteworthy flags:

1. The `-k` flag can be used to only run tests for which part of the name matches a specified string
2. The `-s` flag is extremely useful when debugging.
   By default, `pytest` captures and suppresses data sent to stdout (e.g. with `print`).
   The `-s` flag disables this default behavior (in other words, you can see stuff get printed to the terminal as a test runs).

You can find additional information about invoking pytest [here](https://docs.pytest.org/en/stable/how-to/usage.html) and [here](https://docs.pytest.org/en/stable/reference/reference.html#command-line-flags).

