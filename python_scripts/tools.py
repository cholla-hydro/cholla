import os, sys
from os import listdir
from os.path import isfile, join
import numpy as np


def create_directory( dir ):
  print "Creating Directory: ", dir
  indx = dir[:-1].rfind('/' )
  inDir = dir[:indx]
  dirName = dir[indx:].replace('/','')
  dir_list = next(os.walk(inDir))[1]
  if dirName in dir_list: print " Directory exists"
  else:
    os.mkdir( dir )
    print " Directory created"

def get_files_names( fileKey, inDir, type='cholla' ):
  if type=='nyx': dataFiles = [f for f in listdir(inDir) if (f.find(fileKey) >= 0 )  ]
  if type == 'cholla': [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find(fileKey) > 0 ) ) ]
  dataFiles = np.sort( dataFiles )
  nFiles = len( dataFiles )
  # index_stride = int(dataFiles[1][len(fileKey):]) - int(dataFiles[0][len(fileKey):])
  if type == 'nyx': return dataFiles, nFiles
  if type == 'cholla': return dataFiles, nFiles
