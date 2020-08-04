#/bin/bash

case $1 in 
  hydro)
    #-- find the last file
    output=$(find -name '*.txt.0' | sort | tail -n 1)
    if diff -q  $output $2; then
      echo "$1: PASSED"
    else 
      echo "$1: FAILED"
    fi

esac
