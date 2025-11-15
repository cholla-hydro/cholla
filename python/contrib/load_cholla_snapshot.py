import h5py as h5
import numpy as np

data_directory = '/gpfs/alpine/proj-shared/ast149/cosmo_tests/cosmo_1024/'
input_directory = data_directory + 'snapshots/'

n_snapshot = 310

particles_file_name = 'particles_{0:03}.h5'.format( n_snapshot )
particles_data = h5.File( input_directory + particles_file_name, 'r')
current_z = particles_data.attrs['current_z']
dm_density = particles_data['density'][...] 
potential = particles_data['grav_potential'][...]


grid_file_name = 'grid_{0:03}.h5'.format( n_snapshot )
grid_data = h5.File( input_directory + grid_file_name, 'r')
gas_density = grid_data['density'][...] 
HI_density = grid_data['HI_density'][...] 
temperature = grid_data['temperature'][...]

print( dm_density.mean() )
print( potential.mean() )
print( gas_density.mean() )
print( HI_density.mean() )
print( temperature.mean() )