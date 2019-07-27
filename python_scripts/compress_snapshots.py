import os, sys
from os import listdir
from os.path import isfile, join
from data_compress_grid import compress_grid
from data_compress_particles import compress_particles
from tools import create_directory
import numpy as np



# dataDir = '/gpfs/alpine/proj-shared/ast149/cosmo_tests/'
# inDir = dataDir + 'cosmo_1024/output_snapshots/'
# outDir = dataDir + 'cosmo_1024/snapshots/'


dataDir = '/raid/bruno/data/'
inDir = dataDir + 'cosmo_sims/cholla_pm/256_cool_uv_100Mpc/'
outDir = inDir + 'data_100/'

hydro = True
cosmo = True


def split_name( file_name):
  nSap, name, nBox = file_name.split('.')
  return [int(nSap), int(nBox)]


print( 'Input Dir: ' + inDir )
print( 'Output Dir: ' + outDir )
# create_directory( outDir )
print("")

name_base = 'h5'
dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find('.h5.') > 0 ) and ( f.find('_particles') < 0) ) ]
dataFiles = np.sort( dataFiles )
nFiles = len( dataFiles )

files_names = np.array([ split_name( file_name ) for file_name in dataFiles ])
snaps, boxes = files_names.T
snapshots_all = np.unique( snaps )
boxes = np.unique( boxes )
snapshots_all.sort()
nSnapshots = len( snapshots_all )
nBoxes = len( boxes )

print( "Number of snapshots: {0}".format(nSnapshots) )
print( "Number of files per snapshot: {0}".format(nBoxes) )


#Set wich snapshots to compress
snapshots_to_compress = snapshots_all
snapshots_to_compress = [0]


print( "\nNumber of snapshots to compres: {0}".format(len(snapshots_to_compress)) )


#available Hydro Fields:
#[ density, momentum_x, momentum_y, momentum_z, Enegy, GasEnergy ]
#[ HI_density, HI_density, HeI_density, HeII_density, HeIII_density, e_density, metal_density, temperature, potential ]
hydro_fields = 'all'
hydro_fields = ['density', 'temperature']
print( "\nHydro fields: {0}".format(hydro_fields))

#available Particles Fields:
#[ density, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, particle_IDs ]
particles_fields = 'all'
particles_fields = ['density']
print( "\nParticles fields: {0}".format(particles_fields))


precision = np.float64
# precision = np.float32
# precision = np.float16
print( "\nPrecision: {0}".format( precision ))




print( "\nCompressing Snapshots..." )
for nSnap in snapshots_to_compress:
  if hydro:
    out_base_name = 'grid_' 
    compress_grid( nSnap, nBoxes, name_base, out_base_name, inDir, outDir, hydro_fields,  precision=precision )
  if cosmo or particles:
    out_base_name = 'particles_' 
    compress_particles( nSnap, nBoxes, name_base, out_base_name, inDir, outDir, particles_fields, precision=precision )