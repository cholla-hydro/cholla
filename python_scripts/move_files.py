import os, sys  
import h5py as h5


dataDir = '/gpfs/alpine/proj-shared/ast149/cosmo_tests/cosmo_1024/'
snapDir = dataDir + 'output_snapshots/'

nSnap = 0

file_name = 'grid_{0}.h5'.format( nSnap )
file = h5.File( snapDir + file_name, 'r' )
current_z = file.attrs['current_z']
file.close()

print( ' Snap: {0}   current_z: {1:.4f}'.format( nSnap, current_z) ) 

# os.rename('guru99.txt','career.guru99.txt') 