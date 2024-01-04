FROM rocm/dev-ubuntu-20.04:5.2.3

# Avoid annoying cmake -> tzdata install prompt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install \
    cmake \
    git \
    gnupg \
    hipfft \
    libgtest-dev \
    libhdf5-serial-dev \
    libopenmpi-dev \
    openmpi-bin \
    rocfft \
    software-properties-common \
    wget

# Needed to trick ROCm into thinking there's a GPU
RUN echo "gfx90a" | sudo tee --append $(hipconfig -R)/bin/target.lst

# Install rocRand
RUN apt-get -y install rocrand

# Install Clang and Tools
# RUN wget https://apt.llvm.org/llvm.sh && \
#     chmod +x llvm.sh && \
#     echo "\n" | ./llvm.sh 15 all && \
#     find /usr/bin/ -name 'clang*15'  | sed -E 's/^(\/usr\/bin\/.*)(\-[0-9]*)$/ln -s -v \1\2 \1/' | xargs -d '\n' -n 1 bash -c

# Needed by Cholla Makefile
ENV CHOLLA_MACHINE=github
ENV HIPCONFIG=/opt/rocm-5.2.3
ENV ROCM_PATH=/opt/rocm-5.2.3
ENV HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial
ENV MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
