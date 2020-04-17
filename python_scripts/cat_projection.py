#!/usr/bin/env python3
# Example file for concatenating on-axis projection data
# created when the -DPROJECTION flag is turned on

import h5py
import numpy as np

ns = 0
ne = 0
n_procs = 16 # number of processors that did the cholla calculation
dnamein = './hdf5/raw/'
dnameout = './hdf5/'

# loop over the output times
for n in range(ns, ne+1):

  # open the output file for writing
  fileout = h5py.File(dnameout+str(n)+'_proj.h5', 'w')

  # loop over files for a given output time
  for i in range(0, n_procs):

    # open the input file for reading
    filein = h5py.File(dnamein+str(n)+'_proj.h5.'+str(i), 'r')
    # read in the header data from the input file
    head = filein.attrs

    # if it's the first input file, write the header attributes
    # and create the datasets in the output file
    if (i == 0):
      nx = head['dims'][0]
      ny = head['dims'][1]
      nz = head['dims'][2]
      fileout.attrs['dims'] = [nx, ny, nz]
      fileout.attrs['gamma'] = [head['gamma'][0]]
      fileout.attrs['t'] = [head['t'][0]]
      fileout.attrs['dt'] = [head['dt'][0]]
      fileout.attrs['n_step'] = [head['n_step'][0]]

      dxy = np.zeros((nx,ny))
      dxz = np.zeros((nx,nz))
      Txy = np.zeros((nx,ny))
      Txz = np.zeros((nx,nz))

    # write data from individual processor file to
    # correct location in concatenated file
    nxl = head['dims_local'][0]
    nyl = head['dims_local'][1]
    nzl = head['dims_local'][2]
    xs = head['offset'][0]
    ys = head['offset'][1]
    zs = head['offset'][2]

    dxy[xs:xs+nxl,ys:ys+nyl] += filein['d_xy']
    dxz[xs:xs+nxl,zs:zs+nzl] += filein['d_xz']
    Txy[xs:xs+nxl,ys:ys+nyl] += filein['T_xy']
    Txz[xs:xs+nxl,zs:zs+nzl] += filein['T_xz']

    filein.close()

  # write out the new datasets
  fileout.create_dataset('d_xy', data=dxy)
  fileout.create_dataset('d_xz', data=dxz)
  fileout.create_dataset('T_xy', data=Txy)
  fileout.create_dataset('T_xz', data=Txz)

  fileout.close()
