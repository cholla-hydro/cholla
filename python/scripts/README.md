# Overview

This directory contains some useful scripts for working with Cholla datasets

## Importing from Scripts

Some of the scripts provide advice for modifying ``sys.path`` in order to import functionality. It would be better to move any functions that people want to import directly within the ``cholla_utils``. With that said, it's unclear (at the time of writing) whether anybody actually imports from scripts (concatenation is much less important than it used to be).

If you are somebody that commonly imports from scripts, please open GitHub issue to let us know. (If you're feeling more ambitious, we would also welcome a PR where you add the desired functionality into the ``cholla_utils``.

## Embedding CLI into Module

There may be some value to embedding (some of) this functionality directly within the ``cholla_utils`` module, so that once the module is installed, the functionality can be accessed from anywhere by invoking ``python -m cholla_utils <subcommand> <args...>`` from the command-line. But again
