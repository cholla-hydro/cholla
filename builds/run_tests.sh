#!/usr/bin/env bash

# Builds and runs tests with a given make type. Note that the user still has to
# source the appropriate setup script.
#
# External Dependencies:
#   - Googletest
#   - Cholla dependencies
#   - Bash 4.0 or greater


#set -x #echo all commands

# ==============================================================================
# Perform all the setup required for testing with Cholla.
#
# \param[in] -t (optional) The make type, defaults to hydro
# \param[in] -c (optional) The compiler to use/setup file partial name. The
# setup scripts are all named like setup.MACHINE_NAME.COMPILER.sh and this
# argument is the value of COMPILER which does not occur for all setup scripts
setupTests ()
{
  echo -e "\nRunning Setup..."
  unset CHOLLA_MAKE_TYPE
  unset CHOLLA_COMPILER
  unset MAKE_TYPE_ARG
  # Check arguments & default CHOLLA_MAKE_TYPE
  export CHOLLA_MAKE_TYPE='hydro'

  local OPTIND
  while getopts "t:c:" opt; do
    case $opt in
        t)  # Set the make type
            export CHOLLA_MAKE_TYPE="${OPTARG}"
            ;;
        c)  # Choose the compiler
            CHOLLA_COMPILER=".${OPTARG}"
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            return 1
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            return 1
            ;;
    esac
  done

  # Get the full path to Cholla
  export CHOLLA_ROOT="$(git rev-parse --show-toplevel)"
  if [[ CHOLLA_ROOT == *cholla ]]; then
    echo "Please call this function from within the Cholla repo"
    return 1
  fi

  # Determine the hostname then use that to pick the right machine name and launch
  # command
  if [[ -n ${CHOLLA_MACHINE+x} ]]; then
    FQDN=$CHOLLA_MACHINE
  else
    FQDN=$(hostname --fqdn)
  fi

  case $FQDN in
    *summit* | *peak*)
      echo "summit"
      export CHOLLA_MACHINE='summit'
      export CHOLLA_LAUNCH_COMMAND=("jsrun --smpiargs=\"-gpu\" --cpu_per_rs 1 --tasks_per_rs 1 --gpu_per_rs 1 --nrs")
      ;;
    *crc*)
      export CHOLLA_MACHINE='crc'
      export CHOLLA_LAUNCH_COMMAND='mpirun -np'
      ;;
    *spock*)
      export CHOLLA_MACHINE='spock'
      export CHOLLA_LAUNCH_COMMAND='srun -n'
      ;;
    *c3po*)
      export CHOLLA_MACHINE='c3po'
      export CHOLLA_LAUNCH_COMMAND='mpirun -np'
      ;;
    *frontier* | *crusher*)
      export CHOLLA_MACHINE='frontier'
      export CHOLLA_LAUNCH_COMMAND='srun -n'
      ;;
    *github*)
      export CHOLLA_MACHINE='github'
      export CHOLLA_LAUNCH_COMMAND='mpirun -np'
      ;;
    *)
      echo "No settings were found for this host. Current host: ${FQDN}" >&2
      return 1
      ;;
  esac

  # Clean the cholla directory
  builtin cd $CHOLLA_ROOT
  make clobber

  # Source the setup file
  source "${CHOLLA_ROOT}/builds/setup.${CHOLLA_MACHINE}${CHOLLA_COMPILER}.sh"
}
# ==============================================================================

# ==============================================================================
# Build Cholla itself. Requires that the setupTests function has already been
# called
buildCholla ()
{
  echo -e "\nBuilding Cholla...\n"
  builtin cd $CHOLLA_ROOT
  make -j TYPE=${CHOLLA_MAKE_TYPE} BUILD=${1}
}
# ==============================================================================

# ==============================================================================
# Build the Cholla tests. Requires that the setupTests function has already been
# called
buildChollaTests ()
{
  echo
  builtin cd $CHOLLA_ROOT
  make -j TYPE=${CHOLLA_MAKE_TYPE} TEST=true
}
# ==============================================================================

# ==============================================================================
# Build GoogleTest, create "install" directory that mimics a system install
# directory, export GOOGLETEST_ROOT as the path to the install directory
buildGoogleTest ()
{
  echo -e "\nBuilding GoogleTest..."

  builtin cd $CHOLLA_ROOT

  # All the flags to pass to GoogleTest when building and GTEST URL
  GOOGLETEST_URL="https://github.com/google/googletest.git"

  # Uncomment this line to run with the last version of GoogleTest that supports
  # C++11
  # GOOGLETEST_VERSION="-b release-1.12.1"

  CMAKE_FLAGS=(-DGTEST_HAS_PTHREAD=1
              -DCMAKE_C_COMPILER=gcc
              -DCMAKE_CXX_COMPILER=g++)

  # Download and build
  git clone --depth 1 $GOOGLETEST_VERSION $GOOGLETEST_URL  && \
  builtin cd googletest      && \
  mkdir build                && \
  builtin cd build           && \
  cmake .. ${CMAKE_FLAGS[@]} && \
  make -j                    && \

  # Now we "install" it in the googletest/build/install_root directory
  echo -e "\nSetting up install directory for GoogleTest..."

  # Destination directory, full path required
  DEST_DIR="${CHOLLA_ROOT}/googletest/build/install_root" && \

  # Make the required directories
  mkdir "${DEST_DIR}" && \
  mkdir "${DEST_DIR}/include" && \
  mkdir "${DEST_DIR}/lib64" && \

  # Copy include directories
  cp -r "../googletest/include/gtest" "${DEST_DIR}/include/" && \
  cp -r "../googlemock/include/gmock" "${DEST_DIR}/include/" && \

  # Copy lib directory contents
  cp -r lib/* "${DEST_DIR}/lib64/" && \

  export GOOGLETEST_ROOT="${DEST_DIR}"

  builtin cd $CHOLLA_ROOT
}
# ==============================================================================

# ==============================================================================
# Run the tests with the required command line arguments.
# \param[in] $1 (optional) Any additional arguments to pass to the executable,
# usually gtest command line flags
runTests ()
{
  echo -e "\nRunning Tests...\n"

  # Determine paths and set launch flags
  CHOLLA_OPTIONS=("--cholla-root" "${CHOLLA_ROOT}"
                  "--build-type" "${CHOLLA_MAKE_TYPE}"
                  "--machine" "${CHOLLA_MACHINE}"
                  "--mpi-launcher" "${CHOLLA_LAUNCH_COMMAND[@]}")

  GTEST_FILTER="--gtest_filter=*tALL*:*t${CHOLLA_MAKE_TYPE^^}*"
  GTEST_REPORT="--gtest_output=xml:${CHOLLA_ROOT}/bin/"

  builtin cd $CHOLLA_ROOT
  set -x
  "${CHOLLA_ROOT}/bin/cholla.${CHOLLA_MAKE_TYPE}.${CHOLLA_MACHINE}.tests" \
  "${CHOLLA_OPTIONS[@]}" \
  "${GTEST_FILTER}" \
  "${GTEST_REPORT}" \
  "${@}"
  set +x
}
# ==============================================================================

# ==============================================================================
# Call all the functions required for setting up, building, and running tests
#
# \param[in] -t (optional) The make type, defaults to hydro
# \param[in] -c (optional) The compiler to use/setup file partial name. The
# setup scripts are all named like setup.MACHINE_NAME.COMPILER.sh and this
# argument is the value of COMPILER which does not occur for all setup scripts
# \param[in] -g (optional) If set then download and build a local version of
# GoogleTest to use instead of the machine default
buildAndRunTests ()
{
  # Unset BUILD_GTEST so that subsequent runs aren't tied to what previous runs
  # did
  unset BUILD_GTEST

  BUILD_MODE='OPTIMIZE'

  # Check arguments
  local OPTIND
  while getopts "t:c:g:d" opt; do
    case $opt in
        t)  # Set the make type
            MAKE_TYPE_ARG="-t ${OPTARG}"
            ;;
        c)  # Choose the compiler
            COMPILER_ARG="-c ${OPTARG}"
            ;;
        g)  # Build GoogleTest locally?
            BUILD_GTEST=true
            ;;
        d)  # Build the debug version of Cholla?
            BUILD_MODE='DEBUG'
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            return 1
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            return 1
            ;;
    esac
  done

  # Now we get to setting up and building
  setupTests $MAKE_TYPE_ARG $COMPILER_ARG && \
  if [[ -n $BUILD_GTEST ]]; then
    buildGoogleTest
  fi
  buildCholla $BUILD_MODE && \
  buildChollaTests && \
  runTests
}
# ==============================================================================
