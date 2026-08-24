
# The version of ubuntu in the base image was picked for consistency
# with the ubuntu version in the base image of the cuda image
FROM rocm/dev-ubuntu-22.04:5.5.1

RUN apt-get -y update && apt-get -y install \
    cmake \
    git \
    git-lfs \
    gnupg \
    hipfft \
    libgtest-dev \
    libhdf5-serial-dev \
    libopenmpi-dev \
    openmpi-bin \
    rocfft \
    software-properties-common \
    wget \
    sudo

# Initialize Git LFS configurations (only needed for dev-containters)
RUN git lfs install

# Needed to trick ROCm into thinking there's a GPU
RUN echo "gfx90a" | sudo tee --append $(hipconfig -R)/bin/target.lst

# Install rocRand
RUN apt-get -y install rocrand

## Install Clang and Tools
ADD scripts /tmp/scripts

# this ARG must be declared after the FROM command to perform substitution
ARG TARGET_LLVM_VERSION="17"
RUN python3 /tmp/scripts/get_clang_tidy.py --llvm-version=${TARGET_LLVM_VERSION}

# Needed by Cholla Makefile
ENV CHOLLA_MACHINE=github
ENV HIPCONFIG=/opt/rocm-5.5.1
ENV ROCM_PATH=/opt/rocm-5.5.1
ENV HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial
ENV MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi

# inspired by https://hub.docker.com/_/ubuntu#locales
# -> this is useful for employing this image as a dev container
ENV LANG=C.UTF-8
