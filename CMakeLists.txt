cmake_minimum_required(VERSION 2.8)

project(cholla)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")

# computing options must be turned off on the command line
# to modify default behavior
option(WITH_CUDA "Use GPU" ON)
option(WITH_MPI "Use MPI" ON)
option(WITH_HDF5 "Use HDF5" ON)
option(ERROR_CHECK "Cuda error checking" OFF)
if(ERROR_CHECK)
  add_definitions(-DCUDA_ERROR_CHECK)
endif(ERROR_CHECK)

# physics options set here
#add_definitions(-DPRECISION=1)
add_definitions(-DPRECISION=2)
#add_definitions(-DPCM)
#add_definitions(-DPLMC)
#add_definitions(-DPLMP)
#add_definitions(-DPPMC)
add_definitions(-DPPMP)
#add_definitions(-DEXACT)
add_definitions(-DROE)
#add_definitions(-DDE)
#add_definitions(-DCOOLING_GPU)
#add_definitions(-DCOOLING_CPU)
#add_definitions(-DH_CORRECTION)

file(GLOB_RECURSE SRC_FILES "src/*.c" "src/*.cpp")
file(GLOB_RECURSE CUDA_SRC_FILES "src/*.cu")


if(WITH_CUDA)
  find_package(CUDA CONFIG REQUIRED)
  add_definitions(-DCUDA)
  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  set(CUDA_NVCC_FLAGS -fmad=false)
  cuda_add_library(cudawrap ${CUDA_SRC_FILES}) 
endif(WITH_CUDA)
if(WITH_MPI)
  find_package(MPI CONFIG REQUIRED)
  add_definitions(-DMPI_CHOLLA)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${MPI_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${MPI_LIBRARIES})
endif(WITH_MPI)
if(WITH_HDF5)
  find_package(HDF5 CONFIG REQUIRED)
  add_definitions(-DHDF5)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${HDF5_LIBRARIES})
endif(WITH_HDF5)
find_package(GSL CONFIG REQUIRED)

set(INCLUDE_DIRS ${INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${GSL_INCLUDE_DIRS}
)

set(LIBS ${LIBS} 
    ${GSL_LIBRARIES} 
)

include_directories(${INCLUDE_DIRS})


add_executable(cholla ${SRC_FILES})
if(WITH_CUDA)
  target_link_libraries(cholla cudawrap ${LIBS} )
endif(WITH_CUDA)

