import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np

def compress_particles(  nSnap, nBoxes, name_base, out_base_name,inDir, outDir, fields, precision=np.float64):


  inFileName = '{0}_particles.{1}.{2}'.format(nSnap, name_base, 0)
  inFile = h5py.File( inDir + inFileName, 'r')
  head = inFile.attrs
  dims_all = head['dims']
  dims_local = head['dims_local']
  nz, ny, nx = dims_all
  keys_all = inFile.keys()
  # print(keys_all)

  keys_grid = ['density', 'grav_potential']
  keys_parts = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'mass', 'particle_IDs' ]
  
  if fields in ['all', 'All', 'ALL']: keys = keys_all
  else: keys = fields
  

  
  fileName = out_base_name + '{0}.h5'.format( nSnap )
  fileSnap = h5py.File( outDir + fileName, 'w')
  
  
  print('Particles:')
  print( ' snap: {0}  {1}'.format( nSnap, keys ))
  if inFile.attrs.get('current_z') is not None: print( ' current_z: {0}'.format( inFile.attrs['current_z'][0] ))
  inFile.close()
  
  added_header = False
  
  for key in keys:
    if key not in keys_all:
      print("ERROR key {0} not found".format(key) )
      print(" Availbale keys {0} ".format(keys_all) )
      continue
    print( 'Loading: {0}').format( key )
    data_all = np.zeros( dims_all, dtype=precision )
    data_all_parts = []
    for nBox in range(nBoxes):
      inFileName = '{0}_particles.{1}.{2}'.format(nSnap, name_base, nBox)
      inFile = h5py.File( inDir + inFileName, 'r')
      head = inFile.attrs
      if added_header == False:
        for h_key in head.keys():
          if h_key in ['dims', 'dims_local', 'offset', 'bounds', 'domain', 'dx', ]: continue
          # print h_key
          fileSnap.attrs[h_key] = head[h_key][0]
        added_header = True
  
      procStart_z, procStart_y, procStart_x = head['offset']
      procEnd_z, procEnd_y, procEnd_x = head['offset'] + head['dims_local']
      if key in keys_grid:
        data_local = inFile[key][...]
        data_all[ procStart_z:procEnd_z, procStart_y:procEnd_y, procStart_x:procEnd_x] = data_local
      if key in keys_parts:
        data_local_parts = inFile[key][...]
        data_all_parts.append(data_local_parts)
      inFile.close()
    if key in keys_grid:
      fileSnap.create_dataset( key, data=data_all.astype(precision) )
      fileSnap.attrs['max_'+ key ] = data_all.max()
      fileSnap.attrs['min_'+ key ] = data_all.min()
      fileSnap.attrs['mean_'+ key ] = data_all.mean()
    if key in keys_parts:
      array_parts = np.concatenate(data_all_parts)
      fileSnap.create_dataset( key, data=array_parts.astype(precision) )
      # print 'nParticles: ', len(array_parts)
  
  fileSnap.close()
  print ' Saved File: ', outDir+fileName
