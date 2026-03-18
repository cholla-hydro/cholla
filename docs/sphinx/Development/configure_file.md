(about-configure-file)=
# configure_file.py tool

The {repository-file}`tools/configure_file.py` script is a tool invoked by the build system.

The tool is intended to be used to generate files that are treated as part of the codebase and that hold configuration values set by the build-system.
In short, it acts as an alternative to repeatedly passing long lists of configuration options to the compiler for every source file.

It's instructive to consider the structure of a typical invocation of the tool:

```sh
configure_file.py --input=<template-file> --output=<output> [-D<var>[=<value>]]...
```

The snippet illustrates that the tool recieves 3 kinds of information:
1. The input to the template file read by the tool
2. The path to the output file to be produced by the tool
3. A variable number of options for specifying configuration variables.
   ``-D<var>`` defines a configuration variable ``<var>`` without a value while ``-D<var>=<value>`` associates the variable with the value ``<value>`` 

> Aside: a complete overview of all available options is provided [later](#detailed-configure-file-help)

Essentially, the tool reads template file, performs transformations based upon configuration variables and write out the result.

Analogous functionality is implemented by various build systems such as [autotools](https://www.gnu.org/software/autoconf/manual/autoconf-2.69/html_node/Setting-Output-Variables.html), [CMake](https://cmake.org/cmake/help/latest/command/configure_file.html), and [Meson](https://mesonbuild.com/Configuration.html).
**This tool is closely modelled after the ``confiure_file`` commands provided by CMake and Meson.

## Transformations

The tool performs 2 type of tranformations.

The first transformation type is straightforward **variable substitution**.
The tool replaces tokens enclosed in a pair of ``@`` symbols with the value of the configuration variable matching the token.
The tool aborts with an error if the configuration variable isn't defined or doesn't have an associated value.
Let's consider 2 examples:
1. Consider a template file with the line
   ```c++
   const char* git_hash = "@GIT_HASH@";
   ```
   If the tool is passed `-DGIT_HASH=f4ac1200c2fa21f72a9f160f38f5924f440d19508` the corresponding line in the output file will read:
   ```c++
   const char* git_hash = "f4ac1200c2fa21f72a9f160f38f5924f440d19508";
   ```
2. Consider a template file with the line
   ```c++
   #define PRECISION @PRECISION@
   ```
   If the tool is passed `-DPRECISION=2` the corresponding line in the output file will read:
   ```c++
   #define PRECISION 2
   ```

We refer the second type of transformation as a **define-transfrom**.
This occurs for a line in the input file of the form:
```c++
#configurefile_define CONFVAR
```
where ``CONFVAR`` is a placehold the name of any configuration variable.
The corresponding line of the output file has 3 possible forms:
1. If ``-DCONFVAR`` was passed to the tool, then the output line is
   ```c++
   #define CONFVAR
   ```
2. If ``-DCONFVAR=42`` was passed to the tool, then the output line is:
   ```c++
   #define CONFVAR 42
   ```
   In this scenario, 42 is intended as an arbitrary choice for the associated value
3. If ``CONFVAR`` is not the name of a configuration variable, the output becomes
   ```c++
   /* #undef CONFVAR */
   ```

:::{note}
The analogous transformations in CMake and Meson replace lines that start with ``#cmakedefine`` and ``#mesondefine``.
:::

(detailed-configure-file-help)=
## Detailed Help Output

:::{include-cli-help} ../../tools/configure_file.py
:::