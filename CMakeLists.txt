cmake_minimum_required(VERSION 2.8)

project(cholla)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")

find_package(CUDA)
find_package(MPI)
find_package(GSL)
find_package(HDF5)

#configure_file(config.h.in config.h)
#include_directories(".")

add_definitions(-DCUDA)
add_definitions(-DCUDA_ERROR_CHECK)
add_definitions(-DMPI_CHOLLA)
add_definitions(-DPRECISION=2)
add_definitions(-DROE)
add_definitions(-DPPMP)
add_definitions(-DHDF5)

set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MPI_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
)

set(LIBS ${LIBS} 
    ${CUDA_LIBRARIES}
    ${MPI_LIBRARIES} 
    ${GSL_LIBRARIES} 
    ${HDF5_LIBRARIES}
)

include_directories(${INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS -fmad=false)

file(GLOB_RECURSE SRC_FILES "src/*.c" "src/*.cpp")
file(GLOB_RECURSE CUDA_SRC_FILES "src/*.cu")
cuda_add_library(cudawrap ${CUDA_SRC_FILES}) 


add_executable(cholla ${SRC_FILES})
target_link_libraries(cholla cudawrap ${LIBS} )

