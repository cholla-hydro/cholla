
# notes on the employed image:
# - Needs to be devel (not base or runtime) to have nvcc
# - Ubuntu 22.04 is better than 18 because Ubuntu 22's default git version
#   exceeds 2.17. For context, Github actions requires git > 2.17 so that
#   cholla is pulled into a git repo. Which is required for the Makefile.
#   With ubuntu 22.04 this grabs 2.34.1

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get -y update && apt install -y \
    cmake \
    git \
    git-lfs \
    gnupg \
    libgtest-dev \
    libhdf5-serial-dev \
    libopenmpi-dev \
    openmpi-bin \
    software-properties-common \
    wget \
    sudo

# Initialize Git LFS configurations (only needed for dev-containters)
RUN git lfs install

## Install Clang and Tools
ADD scripts /tmp/scripts

# this ARG must be declared after the FROM command to perform substitution
ARG TARGET_LLVM_VERSION="17"
RUN python3 /tmp/scripts/get_clang_tidy.py --llvm-version=${TARGET_LLVM_VERSION}

# Needed by Cholla Makefile
ENV CHOLLA_MACHINE=github
ENV CUDA_ROOT=/usr/local/cuda-11/
ENV HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial/
ENV MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi/

# inspired by https://hub.docker.com/_/ubuntu#locales
# -> this is useful for employing this image as a dev container
ENV LANG=C.UTF-8
