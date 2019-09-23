import os, sys  
import h5py as h5


dataDir = '/gpfs/alpine/proj-shared/ast149/cosmo_tests/cosmo_1024/'
snapDir = dataDir + 'output_snapshots/'

# nSnap = 1


snaps_found = []
snaps_missing = []
for nSnap in range(30,  80 ):
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



outDir = dataDir + 'snapshots/'

for nSnap in range( 30, 80 ):
  inFile = 'grid_{0}.h5'.format( nSnap )
  outFile = 'grid_{0:03}.h5'.format( nSnap )
  print( '{0}  ->  {1}'.format( inFile, outFile ))
  os.rename( snapDir + inFile, outDir + outFile)
  inFile = 'particles_{0}.h5'.format( nSnap )
  outFile = 'particles_{0:03}.h5'.format( nSnap )
  print( '{0}  ->  {1}'.format( inFile, outFile ))
  os.rename( snapDir + inFile, outDir + outFile)
   