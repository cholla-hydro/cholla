FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# Needs to be devel, not base or runtime, to have nvcc
# Ubuntu 22 is better than 18 because Ubuntu 22 default git is > 2.17
# Github actions requires git > 2.17 so that cholla is pulled into a git repo
# Which is required for the Makefile
# With ubuntu 22.04 this grabs 2.34.1

RUN apt-get -y update && apt install -y \
    cmake \
    git \
    gnupg \
    libgtest-dev \
    libhdf5-serial-dev \
    libopenmpi-dev \
    openmpi-bin \
    software-properties-common \
    wget

# Install Clang and Tools
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    echo "\n" | ./llvm.sh 15 all && \
    find /usr/bin/ -name 'clang*15'  | sed -E 's/^(\/usr\/bin\/.*)(\-[0-9]*)$/ln -s -v \1\2 \1/' | xargs -d '\n' -n 1 bash -c

# Needed by Cholla Makefile
ENV CHOLLA_MACHINE=github
ENV CUDA_ROOT=/usr/local/cuda-11/
ENV HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial/
ENV MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi/
