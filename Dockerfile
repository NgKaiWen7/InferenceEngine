FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    ccache \
    python3 \
    python3-pip \
    unzip \
    zip \
    file \
    gdb \
    clang \
    clang-format \
    lld \
    llvm \
    && rm -rf /var/lib/apt/lists/*

#
# Cross compilation toolchains
#

RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    && rm -rf /var/lib/apt/lists/*

#
# Optional performance libraries
#

RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

#
# Working directory
#

WORKDIR /workspace

CMD ["/bin/bash"]