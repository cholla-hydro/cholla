import numpy as np
import sys, argparse

args = argparse.ArgumentParser \
  ( description = "Compares FILE1 and FILE2 numerically \
      by calculating L1 norms column-wise between the two input files. \
      FILE1 and FILE2 must have the same number of values. FILE1 and FILE2 \
      are the same if the norm of every column is below TOL.", \
    epilog = "Exit status is 0 if files are the same, 1 if different, \
      2 if trouble." )
      
args.add_argument ( "FILE1", metavar = "FILE1", action = "store" )
args.add_argument ( "FILE2", metavar = "FILE2", action = "store" )

args.add_argument \
  ( "--skip", dest = "skip", action = "store", \
    help = "Skip reading the first SKIP lines." )
args.add_argument \
  ( "--prec", dest = "prec", action = "store", \
    help = "Precision to use, either 'single' or 'double' (default)." )
args.add_argument \
  ( "--tol", dest = "tol", action = "store", \
    help = "The norm above which the values are considered different. \
            Default is 10 times machine epsilon for PREC." )

args = args.parse_args()

FileStart = args.FILE1 
FileEnd   = args.FILE2

Skip = 0
if ( args.skip ) : Skip = int ( args.skip )

Precision = 'double'
if ( args.prec) : Precision = args.prec

Tolerance = np.finfo(Precision).eps * 10.0
if ( args.tol ) : Tolerance = float ( args.tol )

try: 

  Start = np.loadtxt ( FileStart, dtype = Precision, skiprows = Skip )
  End   = np.loadtxt ( FileEnd, dtype = Precision, skiprows = Skip )

  DistanceSum = np.abs(End - Start).sum(axis = 0)
  OriginSum = np.abs(Start).sum(axis=0)
  print("OriginSum:", OriginSum)
  print("DistanceSum:", DistanceSum)
  L1_Error = np.where ( OriginSum > 0.0, DistanceSum / OriginSum, DistanceSum )

  if ( np.all ( L1_Error <= Tolerance ) ):
    sys.exit(0)
  else:
    print ( "Tolerance : %E" % Tolerance )
    print ( "L1 norms  :", L1_Error )
    sys.exit(1)

except Exception as e:
  print(e);
  sys.exit(2)
