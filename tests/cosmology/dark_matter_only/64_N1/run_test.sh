#!/bin/bash

echo "Running Cosmology Dark Matter Only 64^3 test"

cholla_root=$(eval "git rev-parse --show-toplevel") 
cd $cholla_root
test_dir=${cholla_root}/tests/cosmology/dark_matter_only/64_N1
echo "Cholla directory: $cholla_root"
echo "Test directory: $test_dir"

if [[ -z "${CHOLLA_ENVSET}" ]]; then
  echo "ERROR: The Cholla environment has not been set for your system."
  echo "This is usually made by running: source builds/setup.<your_system> "
  echo "Make sure this is done before building Cholla "
  exit 1
fi

# Get the MACHINE name
MACHINE=$(sh $cholla_root/builds/machine.sh)
echo "Machine: $MACHINE"

# Build Cholla cosmology dark-matter only 
BUILD_CMD="DFLAGS=-DONLY_PARTICLES make TYPE=cosmology -j"
echo "Building Cholla Cosmology DM-Only: $BUILD_CMD"
eval $BUILD_CMD

cd $test_dir
CHOLLA_EXEC=${cholla_root}/bin/cholla.cosmology.${MACHINE}
if [[ -f $CHOLLA_EXEC ]]; then
  echo "Found Cholla binary: $CHOLLA_EXEC"
else
  echo "ERROR Cholla binary not found. $CHOLLA_EXEC"
  exit 1
fi

# Download the initial conditions
ics_dir=$test_dir/ics_64_50Mpc_dmo/N1_z100
ics_file=$ics_dir/0_particles.h5.0
if [[ -f $ics_file ]]; then
  echo "Found initial conditions file: $ics_file"
else
  mkdir -p $ics_dir
  echo "Downloading intial conditions file..."
  wget https://www.dropbox.com/scl/fi/25ka9c9kc0csxybwy4gww/0_particles.h5.0?rlkey=cws8u6wlcfo0e53vrxd6k2ch6 -O $ics_file 
  if [[ -f $ics_file ]]; then
    echo "Found initial conditions file: $ics_file"
  else
    echo "ERROR: Initial conditions file wans't dowloaded succesfully."
  fi
fi

# Create the output directory
if [[ ! -d snapshot_files ]]; then
  mkdir snapshot_files
fi

# Replave the TEST_DIR location in the parameter file
REPLACE_CMD="sed -i -e 's@TEST_DIR@'"${test_dir}"'@g' parameter_file.txt"
echo "Replicng TEST_DIR: $REPLACE_CMD"
eval $REPLACE_CMD

# Run the simulation
cp $CHOLLA_EXEC .
RUN_CMD="./cholla.cosmology.${MACHINE} parameter_file.txt > simulation_output.txt"
echo "Running test: $RUN_CMD"
eval $RUN_CMD

# # Download the reference snapshot
reference_dir=$test_dir/reference_64_50Mpc_dmo/N1_z0
reference_file=$reference_dir/1_particles.h5.0
if [[ -f $reference_file ]]; then
  echo "Found reference snapshot: $reference_file"
else
  mkdir -p $reference_dir
  echo "Downloading reference snapshot file..."
  wget https://www.dropbox.com/scl/fi/t7go9m3s1jwjnm9p2wkf4/1_particles.h5.0?rlkey=5of9te4qghtm7ge7l1ahldovg -O $reference_file
  if [[ -f $reference_file ]]; then
    echo "Found reference snapshot file: $reference_file"
  else
    echo "ERROR: Reference snapshot file wans't dowloaded succesfully."
  fi
fi

# Compare output against the reference snapshot
VALIDATION_CMD="python $cholla_root/python_scripts/compare_snapshots.py --type particles --snap_id 1  --fields density --dir_0 $reference_dir --dir_1 $test_dir/snapshot_files --tolerance 1e-8"
echo "Validating: $VALIDATION_CMD"
eval $VALIDATION_CMD
# Collace the output of the python script and evaluate if test passed
validation_result=$?                                                           
if [[ $validation_result == 0 ]]; then
  echo "TEST PASSED"
  exit 0
else  
  echo "TEST FAILED"
  exit 1
fi  


