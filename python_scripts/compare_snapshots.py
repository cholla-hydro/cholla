import sys
import h5py as h5
import numpy as np
import argparse
from io_tools import load_cholla_snapshot_distributed

CLI=argparse.ArgumentParser()
CLI.add_argument( "--type", action='store', dest='type', default='hydro')
CLI.add_argument( "--snap_id", type=int, action='store', dest='snap_id', default=0 )
CLI.add_argument( "--snap_id_0", type=int, action='store', dest='snap_id_0', default=None )
CLI.add_argument( "--snap_id_1", type=int, action='store', dest='snap_id_1', default=None )
CLI.add_argument( "--dir_0", action='store', dest='dir_0', default='./')
CLI.add_argument( "--dir_1", action='store', dest='dir_1', default='./')
CLI.add_argument( "--fields", nargs="*", type=str, default=[] )
CLI.add_argument( "--tolerance", type=float, action='store', dest='tolerance', default=1e-8)

# parse the command line
args = CLI.parse_args()
data_type = args.type
fields = args.fields
snap_id = args.snap_id
snap_id_0 = args.snap_id_0
snap_id_1 = args.snap_id_1
dir_0 = args.dir_0
dir_1 = args.dir_1
tolerance = args.tolerance

print(f'Data type: {data_type}')
print(f'Snapshot id: {snap_id}')
print(f'Fields: {fields}')
print(f'Input directory 0: {dir_0}')
print(f'Input directory 1: {dir_1}')
print(f'Tolerance: {tolerance}')

# Load first snapshot
data_snap_0 = load_cholla_snapshot_distributed( data_type, fields,  snap_id, dir_0 )

# Load second snapshot
data_snap_1 = load_cholla_snapshot_distributed( data_type, fields,  snap_id, dir_1 ) 

# Compare fields
print( 'Comparing fields...')
validation_passed = True
for field in fields:
  data_0 = data_snap_0[field]
  data_1 = data_snap_1[field]
  diff = data_1 - data_0
  indices = np.where( data_0 != 0 )
  diff[indices] /= data_0[indices]
  diff = np.abs( diff )
  diff_min = diff.min()
  diff_max = diff.max()
  diff_avg = diff.mean()
  if diff_max < tolerance: pass_text = 'PASSED'
  else:
    pass_text = 'FAILED'
    validation_passed = False  
  print( f'{field:<10} diff.  min: {diff_min:.2e}  max: {diff_max:.2e}  avg: {diff_avg:.2e}  {pass_text}'      )
  
if validation_passed: 
  print('VALIDATION PASSED')
  sys.exit(0)
else:
  print('VALIDATION FAILED')  
  sys.exit(1)    

   



