#!/usr/bin/env bash

# Builds and runs tests with a given make type. Note that the user still has to
# source the appropriate setup script.
#
# External Dependencies:
#   - Googletest
#   - Cholla dependencies
#   - Bash 4.0 or greater


#set -x #echo all commands

# Check the build type
make_type='hydro'
while getopts ":t:" opt; do
    case $opt in
        t)  # Set the make type
            make_type="${OPTARG}"
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            exit 1
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            exit 1
            ;;
    esac
done
echo "Make Type: ${make_type}"

# Determine the hostname then use that to pick the right machine name and launch
# command
FQDN=$(hostname --fqdn)

case $FQDN in
  *summit* | *peak*)
    echo "summit"
    machine='summit'
    launch_command=(jsrun
                    --smpiargs="-gpu"
                    --nrs 1
                    --cpu_per_rs 1
                    --tasks_per_rs 1
                    --gpu_per_rs 1)
    ;;
  *crc.*)
    machine='crc'
    launch_command=''
    ;;
  *spock*)
    machine='spock'
    launch_command=''
    ;;
  *c3po*)
    machine='c3po'
    launch_command=''
    ;;
  *)
    echo "No settings were found for this host. Current host: ${FQDN}" >&2
    exit 1
    ;;
esac

# Determine paths and set launch flags
cholla_path=$(git rev-parse --show-toplevel)
options=("--cholla-root ${cholla_path}"
         "--build-type ${make_type}"
         "--machine ${machine}"
         "--gtest_filter=*tALL*:*t${make_type^^}*")

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