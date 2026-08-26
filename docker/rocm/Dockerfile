# this is a 2-stage build

ARG SOURCE_IMAGE="rocm/dev-ubuntu-22.04:5.5.1"
# -> we try to match the underlying ubuntu image used in the cuda container

ARG TARGET_LLVM_VERSION="17.0.1"

# =========================
# Stage 1: Build clang-tidy
# =========================
# -> we will build clang-tidy in this stage and then in the next stage, we will
#    copy the files out of here (so that we minimize the size of the resulting
#    image)
FROM ${SOURCE_IMAGE} AS builder

# Install llvm build dependencies.
# -> this dependency list was loosely inspired by the build-defaults inside of
#    llvm's sample dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates cmake 2to3 python-is-python3 subversion ninja-build git && \
    rm -rf /var/lib/apt/lists/*

ADD scripts /tmp/scripts

RUN python /tmp/scripts/get_clang_tidy.py \
    --work-dir=/tmp/clang-build \
    --install-dir=/tmp/clang-install \
    --llvm-version=17.0.1

# ===========================================================
# Stage 2: Actually build the image used for compiling cholla
# ===========================================================
FROM ${SOURCE_IMAGE}


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

# Copy clang installation into this container
# -------------------------------------------
# if we also wanted to install the clang compiler, we should just invoke:
#COPY --from=builder /tmp/clang-install/ /usr/local/

# in the current version, we perform the following
COPY --from=builder /tmp/clang-install/include /usr/local/include
COPY --from=builder /tmp/clang-install/bin/clang-tidy /usr/local/bin/clang-tidy
COPY --from=builder /tmp/clang-install/bin/run-clang-tidy /usr/local/bin/run-clang-tidy

# Needed by Cholla Makefile
ENV CHOLLA_MACHINE=github
ENV HIPCONFIG=/opt/rocm-5.5.1
ENV ROCM_PATH=/opt/rocm-5.5.1
ENV HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial
ENV MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi

# inspired by https://hub.docker.com/_/ubuntu#locales
# -> this is useful for employing this image as a dev container
ENV LANG=C.UTF-8
