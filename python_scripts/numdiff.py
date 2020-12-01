import numpy as np
import sys, argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

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
args.add_argument \
  ( "-q", dest = "quiet", action = "store_true", \
    help = "Print a message only when files differ." )
args.add_argument \
  ( "-v", dest = "verbose", action = "store_true", \
    help = "Always print L1 norms, PREC, and TOL. By default they are only \
            printed when the files are different." )
            
args = args.parse_args()

FileStart = args.FILE1 
FileEnd   = args.FILE2

Skip = 0
if ( args.skip ) : Skip = int ( args.skip )

Precision = 'double'
if ( args.prec) : Precision = args.prec

Tolerance = np.finfo(Precision).eps * 10.0
if ( args.tol ) : Tolerance = float ( args.tol )

Status = -1

try: 

  Start = np.loadtxt ( FileStart, dtype = Precision, skiprows = Skip )
  End   = np.loadtxt ( FileEnd, dtype = Precision, skiprows = Skip )

  DistanceSum = np.abs(End - Start).sum(axis = 0)
  OriginSum = np.abs(Start).sum(axis=0)
  L1_Error = np.where ( OriginSum > 0.0, DistanceSum / OriginSum, DistanceSum )

  if ( np.all ( L1_Error <= Tolerance ) ):
    Status = 0 
  else:
    Status = 1
  
  if ( args.verbose is True ) or ( Status == 1 and not args.quiet ): 
    print ( "Precision : %s" % Precision )
    print ( "Tolerance : %E" % Tolerance )
    print ( "L1 norms  :", L1_Error, "\n" )
  
  if ( Status == 1 ):
    print ( "File %s and %s differ" % (FileStart, FileEnd ) )
  
  sys.exit(Status)

except Exception as e:
  print(e);
  sys.exit(2)
