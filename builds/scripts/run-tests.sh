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
      export CHOLLA_LAUNCH_COMMAND=(jsrun
                                    --smpiargs="-gpu"
                                    --nrs 1
                                    --cpu_per_rs 1
                                    --tasks_per_rs 1
                                    --gpu_per_rs 1)
      ;;
    *crc*)
      export CHOLLA_MACHINE='crc'
      export CHOLLA_LAUNCH_COMMAND=''
      ;;
    *spock*)
      export CHOLLA_MACHINE='spock'
      export CHOLLA_LAUNCH_COMMAND=''
      ;;
    *c3po*)
      export CHOLLA_MACHINE='c3po'
      export CHOLLA_LAUNCH_COMMAND=''
      ;;
    *github*)
      export CHOLLA_MACHINE='github'
      export CHOLLA_LAUNCH_COMMAND=''
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
  make -j TYPE=${CHOLLA_MAKE_TYPE}
}
# ==============================================================================

# ==============================================================================
# Build the Cholla tests. Requires that the setupTests function has already been
# called
buildChollaTests ()
{
  echo -e "\nBuilding Tests...\n"
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
  CMAKE_FLAGS=(-DGTEST_HAS_PTHREAD=1
              -DCMAKE_C_COMPILER=gcc
              -DCMAKE_CXX_COMPILER=g++)

  # Download and build
  git clone $GOOGLETEST_URL  && \
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
# \param[in] $1 (optional) The gtest filter command to use. Defaults to
# "--gtest_filter=*tALL*:*t${CHOLLA_MAKE_TYPE^^}*". $1 must include the entire
# command, not just the arguments, i.e. "--gtest_filter=*PATTERN*" not just
# "*PATTERN*"
runTests ()
{
  echo -e "\nRunning Tests...\n"

  # Determine paths and set launch flags
  CHOLLA_OPTIONS=("--cholla-root ${CHOLLA_ROOT}"
                  "--build-type ${CHOLLA_MAKE_TYPE}"
                  "--machine ${CHOLLA_MACHINE}")

  if [[ -n ${1+x} ]]; then
    GTEST_FILTER="${1}"
  else
    GTEST_FILTER="--gtest_filter=*tALL*:*t${CHOLLA_MAKE_TYPE^^}*"
  fi

  builtin cd $CHOLLA_ROOT
  ${launch_command[@]} \
    ${CHOLLA_ROOT}/bin/cholla.${CHOLLA_MAKE_TYPE}.${CHOLLA_MACHINE}.tests \
    ${CHOLLA_OPTIONS[@]} ${GTEST_FILTER}
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
  # Check arguments
  local OPTIND
  while getopts "t:c:g" opt; do
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
  if [[ BUILD_GTEST ]]; then
    buildGoogleTest
  fi
  buildCholla  && \
  buildChollaTests  && \
  runTests
}
# ==============================================================================
