import h5py
from shutil import copy

# Paths for all the files
rootPath      = ''
hydroPath     = rootPath + '1.h5.0'
particlesPath = rootPath + '1_particles.h5.0'
destPath      = rootPath + 'combined.h5'

# Open the hdf5 files and create the new file
hydroFile     = h5py.File(hydroPath,'r')
particlesFile = h5py.File(particlesPath,'r')

copy(hydroPath, destPath)

destFile      = h5py.File(destPath,'r+')

# Now lets get a list of everything in both source files
hydroAttrKeys        = hydroFile.attrs.keys()
particlesDatasetKeys = particlesFile.keys()
particlesAttrKeys    = particlesFile.attrs.keys()

# Copy all the attributes in the particles file that weren't in the hydro file
for key in particlesAttrKeys:
    if not key in hydroAttrKeys:
        destFile.attrs[key] = particlesFile.attrs[key]

# Now we're going to copy all the datasets from the particles file. Note that
# the "density" dataset requires special care to avoid duplicating names
destFile.copy(particlesFile['density'], 'particle_density')
for key in particlesDatasetKeys:
    if key != 'density':
        destFile.copy(particlesFile[key], key)
