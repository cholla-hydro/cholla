#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.

 A simple template for Dask scripts running on a single machine
================================================================================
"""

import dask
import dask.array as da
import dask.dataframe as dd
from dask import graph_manipulation

import argparse
import pathlib

# ==============================================================================
def main():
    cli = argparse.ArgumentParser()
    # Required Arguments
    # Optional Arguments
    cli.add_argument('-n', '--num-workers', type=int, default=8, help='The number of workers to use.')
    args = cli.parse_args()

    # Set scheduler type. Options are 'threads', 'processes', 'single-threaded', and 'distributed'.
    # - 'threads' uses threads that share memory, often fastest on single machines, can run into issuse with the GIL
    # - 'processes' uses multiple processes that do not share memory, can be used to get around issues with the GIL
    # - `single-threaded` is great for debugging
    dask.config.set(scheduler='processes', num_workers=args.num_workers)

    # Perform your computation
    # ...
    # ...
    # ...
    # Some suggestions:
    # - If you're using Delayed then append all tasks to a list and execute them with `dask.compute(*command_list)`
    # - Visualize task tree with `dask.visualize(*command_list, filename=str('filename.pdf'))
    # - Add dependencies manually with `dask.graph_manipulation.bind(dependent_task, list_of_dependencies)`
    # End of Computation
# ==============================================================================

if __name__ == '__main__':
    from timeit import default_timer
    start = default_timer()
    main()
    print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
