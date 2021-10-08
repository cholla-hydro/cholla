#!/usr/bin/env bash

# Builds and runs tests with a given make type. Note that the user still has to
# source the appropriate setup script.
#
# External Dependencies:
#   - Googletest
#   - Cholla dependencies
#   - Bash 4.0 or greater


#set -x #echo all commands

make_type='hydro'
cholla_path=$(git rev-parse --show-toplevel)

# Summit Settings
#machine='summit'
#launch_command=(jsrun
#                --smpiargs="-gpu"
#                --nrs 1
#                --cpu_per_rs 1
#                --tasks_per_rs 1
#                --gpu_per_rs 1)

# Spock Settings
# machine='spock'
# launch_command=''

# CRC settings
machine='crc'
launch_command=''

options=("--cholla-root ${cholla_path}"
         "--build-type ${make_type}"
         "--machine ${machine}")

cd ${cholla_path}

echo -e "\nCleaning...\n" && \
make clobber > compile.log && \
rm -rf ${cholla_path}/bin/* >> compile.log && \
echo -e "\nBuilding Cholla...\n" && \
make -j TYPE=${make_type}  >> compile.log && \
echo -e "\nBuilding Tests...\n" && \
make -j TYPE=${make_type} TEST=true  >> compile.log && \
echo -e "\nRunning Tests...\n" && \
${launch_command[@]}  ./bin/cholla.${make_type}.${machine}.tests ${options[@]}
