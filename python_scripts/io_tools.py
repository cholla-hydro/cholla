import sys, os
import h5py as h5
import numpy as np


# Open Cholla file. Works with new and original output formatting
def open_cholla_file(  data_type,  input_dir, snapshot_id, box_id ):
  if data_type == 'hydro': base_name = ''
  elif data_type == 'particles': base_name = '_particles'
  elif data_type == 'gravity': base_name = '_gravity'
  else:
    print(f'ERROR: not valid data_type: {data_type} ')
  new_input_dir = input_dir + f'{snapshot_id}/'
  file_name = f'{snapshot_id}{base_name}.h5.{box_id}' 

  # Try loading with the new formatting  
  try:
    input_file = h5.File( new_input_dir + file_name, 'r')
  except:
    found_file = False
  else:
    found_file = True
  if found_file: return input_file

  # Try loading with the original formatting
  try:
    input_file = h5.File( input_dir + file_name, 'r')
  except:
    found_file = False
  else:
    found_file = True
  if found_file: return input_file
  else:
    print( f'File {file_name} not found in input directory: {new_input_dir}')
    print( f'File {file_name} not found in input directory: {input_dir}')
    print( f'ERROR: Unable to load the snapshot file')
    sys.exit()  
  

def get_domain_block( proc_grid, box_bounds, box_size, grid_size ):
  np_x, np_y, np_z = proc_grid
  Lx, Ly, Lz = box_size
  nx_g, ny_g, nz_g = grid_size
  dx, dy, dz = Lx/np_x, Ly/np_y, Lz/np_z
  nx_l, ny_l, nz_l = nx_g//np_x, ny_g//np_y, nz_g//np_z,

  bound_x, bound_y, bound_z = box_bounds

  domain = {}
  domain['global'] = {}
  domain['global']['dx'] = dx
  domain['global']['dy'] = dy
  domain['global']['dz'] = dz
  for k in range(np_z):
    for j in range(np_y):
      for i in range(np_x):
        pId = i + j*np_x + k*np_x*np_y
        domain[pId] = { 'box':{}, 'grid':{} }
        xMin, xMax = bound_x+i*dx, bound_x+(i+1)*dx
        yMin, yMax = bound_y+j*dy, bound_y+(j+1)*dy
        zMin, zMax = bound_z+k*dz, bound_z+(k+1)*dz
        domain[pId]['box']['x'] = [xMin, xMax]
        domain[pId]['box']['y'] = [yMin, yMax]
        domain[pId]['box']['z'] = [zMin, zMax]
        domain[pId]['box']['dx'] = dx
        domain[pId]['box']['dy'] = dy
        domain[pId]['box']['dz'] = dz
        domain[pId]['box']['center_x'] = ( xMin + xMax )/2.
        domain[pId]['box']['center_y'] = ( yMin + yMax )/2.
        domain[pId]['box']['center_z'] = ( zMin + zMax )/2.
        gxMin, gxMax = i*nx_l, (i+1)*nx_l
        gyMin, gyMax = j*ny_l, (j+1)*ny_l
        gzMin, gzMax = k*nz_l, (k+1)*nz_l
        domain[pId]['grid']['x'] = [gxMin, gxMax]
        domain[pId]['grid']['y'] = [gyMin, gyMax]
        domain[pId]['grid']['z'] = [gzMin, gzMax]
  return domain

def select_procid( proc_id, subgrid, domain, ids, ax ):
  domain_l, domain_r = domain
  subgrid_l, subgrid_r = subgrid
  if domain_l <= subgrid_l and domain_r > subgrid_l:
    ids.append(proc_id)
  if domain_l >= subgrid_l and domain_r <= subgrid_r:
    ids.append(proc_id)
  if domain_l < subgrid_r and domain_r >= subgrid_r:
    ids.append(proc_id)




def select_ids_to_load( subgrid, domain, proc_grid ):
  subgrid_x, subgrid_y, subgrid_z = subgrid
  nprocs = proc_grid[0] * proc_grid[1] * proc_grid[2]
  ids_x, ids_y, ids_z = [], [], []
  for proc_id in range(nprocs):
    domain_local = domain[proc_id]
    domain_x = domain_local['grid']['x']
    domain_y = domain_local['grid']['y']
    domain_z = domain_local['grid']['z']
    select_procid( proc_id, subgrid_x, domain_x, ids_x, 'x' )
    select_procid( proc_id, subgrid_y, domain_y, ids_y, 'y' )
    select_procid( proc_id, subgrid_z, domain_z, ids_z, 'z' )
  set_x = set(ids_x)
  set_y = set(ids_y)
  set_z = set(ids_z)
  set_ids = (set_x.intersection(set_y)).intersection(set_z )
  return list(set_ids)


def load_cholla_snapshot_distributed( data_type, fields_to_load,  snapshot_id, input_directory, precision=np.float64, subgrid=None, show_progress=True, print_available_fields=True ):



  if input_directory[-1] != '/': input_directory += '/'
  print( f'Input directory: {input_directory}')

  box_id = 0
  data = open_cholla_file( data_type, input_directory, snapshot_id, box_id )

  domain_bounds = data.attrs['bounds']
  domain_length = data.attrs['domain']
  grid_size = data.attrs['dims']
  proc_grid= data.attrs['nprocs']
  available_fields = data.keys()
  print( f'Domain Bounds: {domain_bounds} \nDomain Length: {domain_length} \nGrid size: {grid_size}\nProcs grid: {proc_grid}')
  if print_available_fields: print( f'Available Fields:  {available_fields}')
  data.close()

  # Get the domain_decomposition
  domain = get_domain_block( proc_grid, domain_bounds, domain_length, grid_size )

  # Find the box ids to load given the subgrid
  if not subgrid:  
    subgrid = [ [0, grid_size[0]], [0, grid_size[1]], [0, grid_size[2]] ]
  else:
    for i,sg in enumerate(subgrid):
      if sg[0] < 0 : 
        print(f'ERROR: subgrid along {i} axis: {sg} is less than zero: Setting to {[0, sg[1]]} ')
        sg[0] = 0
      if sg[1] == -1: sg[1] = grid_size[i] # If -1 is pass as the upper bound, use the grid size for that axis
      if sg[1] < sg[0] :  
        sg[1] = sg[0] + 1
        print(f'ERROR: subgrid along {i} axis: right index is less than left index. Setting to slice of width 1  {sg} ') 
      if sg[1] > grid_size[i]: 
        print(f'ERROR: subgrid along {i} axis: {sg} is larger than the grid size: {grid_size[i]}. Setting to {[sg[0], grid_size[i] ]} ')
    print( f'Loading subgrid: {subgrid}')

  
  ids_to_load = select_ids_to_load( subgrid, domain, proc_grid )

  # print(("Loading Snapshot: {0}".format(nSnap)))
  #Find the boundaries of the volume to load
  domains = { 'x':{'l':[], 'r':[]}, 'y':{'l':[], 'r':[]}, 'z':{'l':[], 'r':[]}, }
  for id in ids_to_load:
    for ax in list(domains.keys()):
      d_l, d_r = domain[id]['grid'][ax]
      domains[ax]['l'].append(d_l)
      domains[ax]['r'].append(d_r)
  boundaries = {}
  for ax in list(domains.keys()):
    boundaries[ax] = [ min(domains[ax]['l']),  max(domains[ax]['r']) ]

  # Get the size of the volume to load
  nx = int(boundaries['x'][1] - boundaries['x'][0])    
  ny = int(boundaries['y'][1] - boundaries['y'][0])    
  nz = int(boundaries['z'][1] - boundaries['z'][0])    

  print( f'Loading {data_type} data:')

  dims_all = [ nx, ny, nz ]
  data_out = {}
  for field in fields_to_load:
    data_particles = False
    if field in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'particle_IDs']: data_particles = True 
    if not data_particles: data_all = np.zeros( dims_all, dtype=precision )
    else: data_all = []
    added_header = False
    n_to_load = len(ids_to_load)

    for i, box_id in enumerate(ids_to_load):
      input_file = open_cholla_file( data_type, input_directory, snapshot_id, box_id )
      head = input_file.attrs
      if not added_header :
        for h_key in list(head.keys()):
          if h_key in ['dims', 'dims_local', 'offset', 'bounds', 'domain', 'dx', ]: continue
          data_out[h_key] = head[h_key][0]
        added_header = True

      if show_progress:
        terminalString  = '\r Loading File: {0}/{1}   {2}  {3}'.format(i+1, n_to_load, data_type, field)
        sys.stdout. write(terminalString)
        sys.stdout.flush() 

      if not data_particles:
        procStart_x, procStart_y, procStart_z = head['offset']
        procEnd_x, procEnd_y, procEnd_z = head['offset'] + head['dims_local']
        # Subtract the offsets
        procStart_x -= boundaries['x'][0]
        procEnd_x   -= boundaries['x'][0]
        procStart_y -= boundaries['y'][0]
        procEnd_y   -= boundaries['y'][0]
        procStart_z -= boundaries['z'][0]
        procEnd_z   -= boundaries['z'][0]
        procStart_x, procEnd_x = int(procStart_x), int(procEnd_x)
        procStart_y, procEnd_y = int(procStart_y), int(procEnd_y)
        procStart_z, procEnd_z = int(procStart_z), int(procEnd_z)
        data_local = input_file[field][...]
        data_all[ procStart_x:procEnd_x, procStart_y:procEnd_y, procStart_z:procEnd_z] = data_local
      else:
        data_local = input_file[field][...]
        data_all.append( data_local )    

    # Trim off the excess data outside of the subgrid:
    if not data_particles:
      trim_x_l = subgrid[0][0] - boundaries['x'][0]
      trim_x_r = boundaries['x'][1] - subgrid[0][1]  
      trim_y_l = subgrid[1][0] - boundaries['y'][0]
      trim_y_r = boundaries['y'][1] - subgrid[1][1]  
      trim_z_l = subgrid[2][0] - boundaries['z'][0]
      trim_z_r = boundaries['z'][1] - subgrid[2][1]  
      trim_x_l, trim_x_r = int(trim_x_l), int(trim_x_r) 
      trim_y_l, trim_y_r = int(trim_y_l), int(trim_y_r) 
      trim_z_l, trim_z_r = int(trim_z_l), int(trim_z_r) 
      data_output = data_all[trim_x_l:nx-trim_x_r, trim_y_l:ny-trim_y_r, trim_z_l:nz-trim_z_r,  ]
      data_out[field] = data_output

    else:
      data_all = np.concatenate( data_all )
      data_out[field] = data_all
      if field == 'particle_IDs': data_out[field] = data_out[field].astype( np.int64 ) 

    if show_progress: print("")
  return data_out