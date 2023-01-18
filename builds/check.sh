#/bin/bash

if [ -z ${PYTHON} ]; then
  PYTHON=$(which python)
fi

case $1 in 
  hydro)
    #-- find the last file
    output=$(find -name '*.txt.0' | sort | tail -n 1)
    CMD="$PYTHON python_scripts/numdiff.py -v --skip 8 --prec single $output $2 "
    echo "Running command: ${CMD}"
    echo ""
    $CMD
    if [ $? -eq 0 ]; then
      echo "$1: PASSED"
      exit 0
    else 
      echo "$1: FAILED"
      exit -1
    fi
    ;;
*)
    echo "\"$1\" is not a valid problem type."
    exit -1
esac
