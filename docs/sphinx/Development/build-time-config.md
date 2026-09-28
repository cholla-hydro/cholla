(dev-build-time-config)=
# Build-Time Configuration

This page provides a developer guide for build-time configuration.
The full list of options can be found {ref}`here <makefile-parameters>`

## How Config Parameters are Communicated to the Source Code

At the time of writing, the build-system tracks configuration options in the ``DFLAGS``  variable.
In more detail:
- as ``make`` parses the makefiles it appends a token encoding each configuration variable's to ``DFLAGS``
- a token of defining a variable, ``<var>``, is specified as either ``-D<var>`` or ``-D<var>=<value>``.
  The first format defines a variable without assigning a value while the second variable assocciates the value ``<value>``.

Originally, the build system made the source code aware of the configuration options by passing all the tokens in ``DFLAGS`` directly as arguments to the compiler.
More recently, we have adopted a new strategy.

At present, the build-system uses the {ref}`configure_file.py tool <about-configure-file>` to communicate the configuration parameters.
Specifically, the build system passes all the tokens in ``DFLAGS`` as arguments to ``configure_file.py`` to generate a file called ``cholla_config.h`` from {repository-file}`src/cholla_config.h.in`.
Then the compiler is passed the option ``-include cholla_config.h``, which tells the compiler to act like ``#include "cholla_config.h"`` is the first line of every source file.

:::{important}
The use of ``-include cholla_config.h`` is something of a short-term hack.
There are some concerns that this may not work on all C++ compilers (it was decided that since it works for ``g++`` and ``clang++``, it's ok for now).

The more robust long-term plan is to make every source/header file that needs a compile-time configuration parameter directly include the ``cholla_config.h`` header or include the ``global/global.h`` header (which itself includes ``cholla_config.h``).
Unfortunately, this is not immediately viable because both:
1. the checks of configuration options are currently ubiquitous (I think almost every single source/header has at least one check).
2. Just about every single check checks whether a configuration option is defined or not.
   Thus, there's a degeneracy between "forgetting to include the ``cholla_config.h`` header and choosing not to enable a feature.
:::

## How to add a new build-time configuration parameter

:::{important}
Before doing anything, ask yourself "can this be a runtime parameter?"
If the answer is "yes" (and it usually is), then sooner or later somebody is going to have to go to the trouble of converting it to a runtime parameter.

Your code contribution may not be merged until you convert it to a runtime parameter.
:::

To add a new parameter, you need to both:
1. add the option to a Makefile
2. add an appropriate line into {repository-file}`src/cholla_config.h.in`

If you forget to do the second step, then the ``configure_file.py`` tool will fail (and the entire build will fail).
This behavior is a FEATURE -- it makes the build fail if an option is mispelled when modifying a Makefile.

### Best Practice

If you find yourself adding a new configuration parameter, you should STRONGLY prefer to configure things so that the configuration parameter is ALWAYS defined and the only thing that changes is its associated value (e.g. 0 or 1).
This generally produces more robust code (and is an important step towards direct inclusion of ``cholla_config.h`` in all files).
