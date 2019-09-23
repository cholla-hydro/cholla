import os, sys  
import h5py as h5


dataDir = '/gpfs/alpine/proj-shared/ast149/cosmo_tests/cosmo_1024/'
snapDir = dataDir + 'output_snapshots/'

nSnap = 150

snaps_found = []
snaps_missing = []

for nSnap in range(1,  100 ):
  file_name = 'grid_{0}.h5'.format( nSnap )
  try:
    file = h5.File( snapDir + file_name, 'r' )
  except IOError, e:
    print( ' Snap: {0}   Not found'.format( nSnap ) ) 
    snaps_missing.append( nSnap )
  else:
    current_z = file.attrs['Current_z']
    print( ' Snap: {0}   current_z: {1:.4f}'.format( nSnap, current_z) ) 
    file.close()
    snaps_found.append( nSnap )

print('Missing: ', snaps_missing)

# os.rename('guru99.txt','career.guru99.txt') 