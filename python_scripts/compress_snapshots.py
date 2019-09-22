import os, sys
from os import listdir
from os.path import isfile, join
from data_compress_grid import compress_grid
from data_compress_particles import compress_particles
from tools import create_directory
import numpy as np
import time


n_arg = len(sys.argv)
if n_arg > 1:
  args = []
  print "Using command arguments"
  for i in range(1 , n_arg):
    arg = sys.argv[i]
    args.append( int( arg ))
  print args
  snap_start, snap_end = args

dataDir = '/gpfs/alpine/proj-shared/ast149/cosmo_tests/cosmo_1024/'
inDir = dataDir + 'output_files/'
outDir = dataDir + 'output_snapshots/'

hydro = True
cosmo = True


def split_name( file_name):
  nSap, name, nBox = file_name.split('.')
  return [int(nSap), int(nBox)]


print( 'Input Dir: ' + inDir )
print( 'Output Dir: ' + outDir )
create_directory( outDir )
print("")

name_base = 'h5'
# dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find('.h5.') > 0 ) and ( f.find('_particles') < 0) ) ]
# dataFiles = np.sort( dataFiles )
# nFiles = len( dataFiles )
# 
# files_names = np.array([ split_name( file_name ) for file_name in dataFiles ])
# snaps, boxes = files_names.T
# snapshots_all = np.unique( snaps )
# boxes = np.unique( boxes )
# snapshots_all.sort()
# nSnapshots = len( snapshots_all )
# nBoxes = len( boxes )

nSnapshots = 310
nBoxes = 512

print( "Number of snapshots: {0}".format(nSnapshots) )
print( "Number of files per snapshot: {0}".format(nBoxes) )


#Set wich snapshots to compress
# snapshots_to_compress = snapshots_all
# snapshots_to_compress = range( 50, 100)
snapshots_to_compress = range( snap_start, snap_end)


print( "\nNumber of snapshots to compres: {0}".format(len(snapshots_to_compress)) )


#available Hydro Fields:
#[ density, momentum_x, momentum_y, momentum_z, Enegy, GasEnergy ]
#[ HI_density, HI_density, HeI_density, HeII_density, HeIII_density, e_density, metal_density, temperature, potential ]
# hydro_fields = 'all'
hydro_fields = ['density', 'temperature', 'HI_density',  ]
print( "\nHydro fields: {0}".format(hydro_fields))

#available Particles Fields:
#[ density, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, particle_IDs ]
# particles_fields = 'all'
particles_fields = ['density', 'grav_potential']
print( "\nParticles fields: {0}".format(particles_fields))


# precision = np.float64
precision = np.float32
# precision = np.float16
print( "\nPrecision: {0}".format( precision ))





print( "\nCompressing Snapshots..." )
for nSnap in snapshots_to_compress:
  start = time.time()
  if hydro:
    out_base_name = 'grid_' 
    compress_grid( nSnap, nBoxes, name_base, out_base_name, inDir, outDir, hydro_fields,  precision=precision )
  if cosmo or particles:
    out_base_name = 'particles_' 
    compress_particles( nSnap, nBoxes, name_base, out_base_name, inDir, outDir, particles_fields, precision=precision )
  end = time.time()
  print( ' Elapsed Time: {0:.2f} min'.format((end - start)/60.) )





























