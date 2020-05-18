#!/usr/bin/env python3
# Example file for concatenating rotated projection data
# created when the -DROTATED_PROJECTION flag is turned on

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
  fileout = h5py.File(dnameout+str(n)+'_rot_proj.h5', 'w')

  # loop over files for a given output time
  for i in range(0, n_procs):

    # open the input file for reading
    filein = h5py.File(dnamein+str(n)+'_rot_proj.h5.'+str(i), 'r')
    # read in the header data from the input file
    head = filein.attrs

    # if it's the first input file, write the header attributes
    # and create the arrays to hold the output data
    if (i == 0):
      nxr = int(head['nxr'])
      nzr = int(head['nzr'])
      Lx = head['Lx']
      Lz = head['Lz']
      delta = head['delta']
      theta = head['theta']
      phi = head['phi']
      gamma = head['gamma']
      t = head['t']
      dt = head['dt']
      n_step = head['n_step']
      fileout.attrs['nxr'] = nxr
      fileout.attrs['nzr'] = nzr
      fileout.attrs['Lx'] = Lx 
      fileout.attrs['Lz'] = Lz 
      fileout.attrs['delta'] = delta 
      fileout.attrs['theta'] = theta 
      fileout.attrs['phi'] = phi
      fileout.attrs['gamma'] = gamma
      fileout.attrs['t'] = t
      fileout.attrs['dt'] = dt
      fileout.attrs['n_step'] = n_step

      d_xzr  = np.zeros((nxr, nzr))
      vx_xzr = np.zeros((nxr, nzr))
      vy_xzr = np.zeros((nxr, nzr))
      vz_xzr = np.zeros((nxr, nzr))
      T_xzr  = np.zeros((nxr, nzr))

    # write data from individual processor file to
    # correct location in concatenated file
    nx_min = int(head['nx_min'])
    nx_max = int(head['nx_max'])
    nz_min = int(head['nz_min'])
    nz_max = int(head['nz_max'])

    d_xzr[nx_min:nx_max,nz_min:nz_max]  += filein['d_xzr'][:]
    vx_xzr[nx_min:nx_max,nz_min:nz_max] += filein['vx_xzr'][:]
    vy_xzr[nx_min:nx_max,nz_min:nz_max] += filein['vy_xzr'][:]
    vz_xzr[nx_min:nx_max,nz_min:nz_max] += filein['vz_xzr'][:]
    T_xzr[nx_min:nx_max,nz_min:nz_max]  += filein['T_xzr'][:]

    filein.close()

  # write out the new datasets
  fileout.create_dataset("d_xzr", data=d_xzr)
  fileout.create_dataset("vx_xzr", data=vx_xzr)
  fileout.create_dataset("vy_xzr", data=vy_xzr)
  fileout.create_dataset("vz_xzr", data=vz_xzr)
  fileout.create_dataset("T_xzr", data=T_xzr)

  fileout.close()
 


