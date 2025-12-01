#!/usr/bin/env python3
"""Python script for merging hydro, particles, and gravity HDF5 files when they're used as fiducial data files for testing

Raises
------
ValueError
    Duplicate datasets in destination and particle data files
ValueError
    Duplicate datasets in destination and gravity data files
"""

import h5py
import pathlib
import argparse
import shutil

# =====================================================================================================================
def main():
    # Initialize the CLI
    cli = argparse.ArgumentParser()

    # Required Arguments
    cli.add_argument('-s', '--source-directory', type=pathlib.Path,  required=True, help='The path to the directory for the source HDF5 files.')
    cli.add_argument('-o', '--output-directory', type=pathlib.Path,  required=True, help='The path to the directory to write out the concatenated HDF5 file.')

    # Optional Arguments
    cli.add_argument('-p', '--particles', default=False, action='store_true', help='')
    cli.add_argument('-g', '--gravity',   default=False, action='store_true', help='')

    # Get the CLI arguments
    args = cli.parse_args()

    # Check that at least one file is being merged
    if not (args.particles or args.gravity):
        cli.error('At least one of the -p/--particles or -g/--gravity arguments are required.')

    # Set the file names
    hydro_path       = args.source_directory / '1.h5.0'
    particle_path    = args.source_directory / '1_particles.h5.0'
    gravity_path     = args.source_directory / '1_gravity.h5.0'
    destination_path = args.output_directory / 'combined.h5'

    # Setup the destination file
    shutil.copy(hydro_path, destination_path)

    # Merge the particle data into the hydro data
    if args.particles:
        merge_particles(particle_path, destination_path)

    # Merge the gravity data into the hydro data
    if args.gravity:
        merge_gravity(gravity_path, destination_path)
# =====================================================================================================================

# =====================================================================================================================
def merge_particles(particle_path: h5py.File, destination_path: pathlib.Path):
    """Merge the particles data file into the destination (i.e. hydro) data file

    Parameters
    ----------
    particle_path : h5py.File
        The path to the source particles file
    destination_path : pathlib.Path
        The path to the destination file

    Raises
    ------
    ValueError
        If a dataset with an identical name exists in both the particles and destination file an error will be raised.
        The only exception to this is the `density` dataset which exists in both files with different meanings, there
        is special handling for that and the particle version is renamed to `particle_density`.
    """
    # Open the files
    particles_file   = h5py.File(particle_path,'r')
    destination_file = h5py.File(destination_path,'r+')

    # Now lets get a list of everything in both source files
    destination_attr_keys = destination_file.attrs.keys()
    destination_data_keys = destination_file.keys()
    particles_attr_keys   = particles_file.attrs.keys()
    particles_data_keys   = particles_file.keys()

    # Copy all the attributes in the particles file that weren't in the destination file
    for key in particles_attr_keys:
        if not key in destination_attr_keys:
            destination_file.attrs[key] = particles_file.attrs[key]

    # Now we're going to copy all the datasets from the particles file. Note that the "density" dataset requires
    # special care to avoid duplicating names
    destination_file.copy(particles_file['density'], 'particle_density')
    for key in particles_data_keys:
        if key != 'density':
            if key not in destination_data_keys:
                destination_file.copy(particles_file[key], key)
            else:
                raise ValueError('Duplicate datasets in destination and particle data files')

    # Close the files
    particles_file.close()
    destination_file.close()
# =====================================================================================================================

# =====================================================================================================================
def merge_gravity(gravity_path: h5py.File, destination_path: pathlib.Path):
    """Merge the gravity data file into the destination (i.e. hydro) data file

    Parameters
    ----------
    gravity_path : h5py.File
        The path to the source gravity file
    destination_path : pathlib.Path
        The path to the destination file

    Raises
    ------
    ValueError
        If a dataset with an identical name exists in both the gravity and destination file an error will be raised
    """
    # Open the files
    gravity_file   = h5py.File(gravity_path,'r')
    destination_file = h5py.File(destination_path,'r+')

    # Now lets get a list of everything in both source files
    destination_attr_keys = destination_file.attrs.keys()
    destination_data_keys = destination_file.keys()
    gravity_attr_keys     = gravity_file.attrs.keys()
    gravity_data_keys     = gravity_file.keys()

    # Copy all the attributes in the particles file that weren't in the destination file
    for key in gravity_attr_keys:
        if not key in destination_attr_keys:
            destination_file.attrs[key] = gravity_file.attrs[key]

    # Now we're going to copy all the datasets from the gravity file.
    for key in gravity_data_keys:
        if key not in destination_data_keys:
            destination_file.copy(gravity_file[key], key)
        else:
            raise ValueError('Duplicate datasets in destination and gravity data files')

    # Close the files
    gravity_file.close()
    destination_file.close()
# =====================================================================================================================

# =====================================================================================================================
if __name__ == '__main__':
    """This just times the execution of the `main()` function
    """
    from timeit import default_timer
    start = default_timer()

    main()

    print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
# =====================================================================================================================