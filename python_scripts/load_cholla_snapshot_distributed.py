import sys
import h5py as h5
import numpy as np
from io_tools import load_cholla_snapshot_distributed

# Script to load Cholla snapshot files without having to merge them into a single file before.
# This is specially useful when dealing with large grids and duplicating the memory footprint 
# of a snapshot becomes inconvenient due to memory and time limitations.    

# The input directory where the snapshots are located 
data_directory = '/lustre/user/bvillase/benchmarcks/cholla/cosmology/256_50Mpc_adiabatic_nmpi8/'
input_directory = data_directory + 'branch_cosmo/snapshot_files/'

# The snapshot id
snapshot_id = 1

# The type of data to load. ['hydro', 'particles', 'gravity']
data_type = 'hydro'

# The fields to load 
fields_to_load = ['density', 'momentum_x', 'Energy']

# Sometimes it is useful to load a subdomain, for example when only a slice of the domain is needed.
# In this case, only the minimum subset of the snapshot files is loaded
# If subgrid = None    The entire domain will be loaded 

subgrid = None # The entire domain will be loaded 

# If subgrid is != None then the subdomain specified by the grid limits will be loaded.
# This is specially useful when dealing with very large snapshots that don't fully fit in memory, 
# and only a slice or sub-volume needs to be loaded   
# Examples:
# Load a sub-volume of 64 x 64 x 64 cells: 
# subgrid = [[0,64], [64,128], [64,128]]

# Load slice of dimensions of one cell width along the x axis full size over the y and z axis: 
# subgrid = [[0,1], [0,-1], [0,-1]]


# Load the data in a specified precision. for example when plotting an image, the data doesn't nit to be
# float64 precision and by lowering to float32 a larger volume can fit in memory   
precision = np.float64
# precision = np.float32

# Load the data
data_hydro = load_cholla_snapshot_distributed(data_type, fields_to_load,  snapshot_id, input_directory, 
                                             precision=precision, subgrid=subgrid )

# Use the loaded fields for whatever you want
gas_density      = data_hydro['density']
gas_total_energy = data_hydro['Energy']


# Load the particles density
fields_to_load = ['density']
data_particles = load_cholla_snapshot_distributed('particles', fields_to_load,  snapshot_id, input_directory, 
                                            precision=precision, subgrid=subgrid )
particles_density = data_particles['density']
