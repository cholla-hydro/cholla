# Example file for concatenating on-axis slice data
# created when the -DSLICES flag is turned on

import h5py
import numpy as np

ns = 0
ne = 0
n_procs = 16 # number of processors that did the cholla calculation
dnamein = './hdf5/raw/'
dnameout = './hdf5/'

DE = False # set to True if Dual Energy flag was used
SCALAR = False # set to True if Scalar was used

# loop over the output times
for n in range(ns, ne+1):

  # open the output file for writing
  fileout = h5py.File(dnameout+str(n)+'_slice.h5', 'w')

  # loop over files for a given output time
  for i in range(0, n_procs):

    # open the input file for reading
    filein = h5py.File(dnamein+str(n)+'_slice.h5.'+str(i), 'r')
    # read in the header data from the input file
    head = filein.attrs

    # if it's the first input file, write the header attributes
    # and create the datasets in the output file
    if (i == 0):
      gamma = head['gamma']
      t = head['t']
      dt = head['dt']
      n_step = head['n_step']
      nx = head['dims'][0]
      ny = head['dims'][1]
      nz = head['dims'][2]
      fileout.attrs['gamma'] = gamma
      fileout.attrs['t'] = t
      fileout.attrs['dt'] = dt
      fileout.attrs['n_step'] = n_step
      fileout.attrs['dims'] = [nx, ny, nz]

      d_xy = np.zeros((nx,ny))
      d_xz = np.zeros((nx,nz))
      d_yz = np.zeros((ny,nz))
      mx_xy = np.zeros((nx,ny))
      mx_xz = np.zeros((nx,nz))
      mx_yz = np.zeros((ny,nz))
      my_xy = np.zeros((nx,ny))
      my_xz = np.zeros((nx,nz))
      my_yz = np.zeros((ny,nz))
      mz_xy = np.zeros((nx,ny))
      mz_xz = np.zeros((nx,nz))
      mz_yz = np.zeros((ny,nz))
      E_xy = np.zeros((nx,ny))
      E_xz = np.zeros((nx,nz))
      E_yz = np.zeros((ny,nz))
      if DE:
       GE_xy = np.zeros((nx,ny))
       GE_xz = np.zeros((nx,nz))
       GE_yz = np.zeros((ny,nz))
      if SCALAR:
       scalar_xy = np.zeros((nx,ny))
       scalar_xz = np.zeros((nx,nz))
       scalar_yz = np.zeros((ny,nz))

    # write data from individual processor file to
    # correct location in concatenated file
    nxl = head['dims_local'][0]
    nyl = head['dims_local'][1]
    nzl = head['dims_local'][2]
    xs = head['offset'][0]
    ys = head['offset'][1]
    zs = head['offset'][2]

    d_xy[xs:xs+nxl,ys:ys+nyl] += filein['d_xy']
    d_xz[xs:xs+nxl,zs:zs+nzl] += filein['d_xz']
    d_yz[ys:ys+nyl,zs:zs+nzl] += filein['d_yz']
    mx_xy[xs:xs+nxl,ys:ys+nyl] += filein['mx_xy']
    mx_xz[xs:xs+nxl,zs:zs+nzl] += filein['mx_xz']
    mx_yz[ys:ys+nyl,zs:zs+nzl] += filein['mx_yz']
    my_xy[xs:xs+nxl,ys:ys+nyl] += filein['my_xy']
    my_xz[xs:xs+nxl,zs:zs+nzl] += filein['my_xz']
    my_yz[ys:ys+nyl,zs:zs+nzl] += filein['my_yz']
    mz_xy[xs:xs+nxl,ys:ys+nyl] += filein['mz_xy']
    mz_xz[xs:xs+nxl,zs:zs+nzl] += filein['mz_xz']
    mz_yz[ys:ys+nyl,zs:zs+nzl] += filein['mz_yz']
    E_xy[xs:xs+nxl,ys:ys+nyl] += filein['E_xy']
    E_xz[xs:xs+nxl,zs:zs+nzl] += filein['E_xz']
    E_yz[ys:ys+nyl,zs:zs+nzl] += filein['E_yz']
    if DE:
      GE_xy[xs:xs+nxl,ys:ys+nyl] += filein['GE_xy']
      GE_xz[xs:xs+nxl,zs:zs+nzl] += filein['GE_xz']
      GE_yz[ys:ys+nyl,zs:zs+nzl] += filein['GE_yz']
    if SCALAR:
      scalar_xy[xs:xs+nxl,ys:ys+nyl] += filein['scalar_xy']
      scalar_xz[xs:xs+nxl,zs:zs+nzl] += filein['scalar_xz']
      scalar_yz[ys:ys+nyl,zs:zs+nzl] += filein['scalar_yz']

    filein.close()

  # wrte out the new datasets
  fileout.create_dataset('d_xy', data=d_xy)
  fileout.create_dataset('d_xz', data=d_xz)
  fileout.create_dataset('d_yz', data=d_yz)
  fileout.create_dataset('mx_xy', data=mx_xy)
  fileout.create_dataset('mx_xz', data=mx_xz)
  fileout.create_dataset('mx_yz', data=mx_yz)
  fileout.create_dataset('my_xy', data=my_xy)
  fileout.create_dataset('my_xz', data=my_xz)
  fileout.create_dataset('my_yz', data=my_yz)
  fileout.create_dataset('mz_xy', data=mz_xy)
  fileout.create_dataset('mz_xz', data=mz_xz)
  fileout.create_dataset('mz_yz', data=mz_yz)
  fileout.create_dataset('E_xy', data=E_xy)
  fileout.create_dataset('E_xz', data=E_xz)
  fileout.create_dataset('E_yz', data=E_yz)
  if DE:
    fileout.create_dataset('GE_xy', data=GE_xy)
    fileout.create_dataset('GE_xz', data=GE_xz)
    fileout.create_dataset('GE_yz', data=GE_yz)
  if SCALAR:
    fileout.create_dataset('scalar_xy', data=scalar_xy)
    fileout.create_dataset('scalar_xz', data=scalar_xz)
    fileout.create_dataset('scalar_yz', data=scalar_yz)

  fileout.close()
