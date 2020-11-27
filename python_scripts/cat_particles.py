# Example file for concatenating particle data

import h5py
import numpy as np

ns = 0
ne = 300
n_procs = 4 # number of processors that did the cholla calculation
dnamein = '/gpfs/alpine/proj-shared/csc380/orlandow/o_cholla/out.21Sep20-Mon-12.49-356588-SOR_ONLY_PARTICLES_DISK/raw/'
dnameout = '/gpfs/alpine/proj-shared/csc380/orlandow/o_cholla/out.21Sep20-Mon-12.49-356588-SOR_ONLY_PARTICLES_DISK/particles_cat/'

# loop over the output times
for n in range(ns, ne+1):

  # open the output file for writing
  fileout = h5py.File(dnameout+str(n)+'_particles.h5', 'w')

  if (n % 10 == 0): print(str(n))

  # loop over files for a given output time
  for i in range(0, n_procs):

    # open the input file for reading
    filein = h5py.File(dnamein+str(n)+'_particles.h5.'+str(i), 'r')
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
      fileout.attrs['velocity_unit'] = head['velocity_unit']
      fileout.attrs['length_unit'] = head['length_unit']
      fileout.attrs['particle_mass'] = head['particle_mass']
      fileout.attrs['density_unit'] = head['density_unit']

      x = np.array([])
      y = np.array([])
      z = np.array([])
      vx = np.array([])
      vy = np.array([])
      vz = np.array([])
      particle_ids = np.array([])
      density = np.zeros((nx, ny, nz))
      n_total_particles = 0


    # write data from individual processor file to
    # correct location in concatenated file
    nxl = head['dims_local'][0]
    nyl = head['dims_local'][1]
    nzl = head['dims_local'][2]
    xs = head['offset'][0]
    ys = head['offset'][1]
    zs = head['offset'][2]

    n_total_particles += head['n_particles_local']
    density[xs:xs+nxl, ys:ys+nyl, zs:zs+nzl] += filein['density']
    x = np.append(x, filein['pos_x'])
    y = np.append(y, filein['pos_y'])
    z = np.append(z, filein['pos_z'])
    vx = np.append(vx, filein['vel_x'])
    vy = np.append(vy, filein['vel_y'])
    vz = np.append(vz, filein['vel_z'])
    particle_ids = np.append(particle_ids, filein['particle_IDs'])

    filein.close()

  # wrte out the new datasets
  fileout.create_dataset('x', data=x)
  fileout.create_dataset('y', data=y)
  fileout.create_dataset('z', data=z)
  fileout.create_dataset('vx', data=vx)
  fileout.create_dataset('vy', data=vy)
  fileout.create_dataset('vz', data=vz)
  fileout.create_dataset('particle_ids', data=particle_ids)
  fileout.create_dataset('density', data=density)
  fileout.attrs['n_total_particles'] = n_total_particles

  fileout.close()
