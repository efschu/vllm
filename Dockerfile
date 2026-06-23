# Dockerfile for vllm GGUF fork with MTP support
# RTX 5090 = Blackwell = sm120, RTX 30 series = sm86
# Only build for needed architectures to reduce memory + time
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
# Only build for sm86 (RTX 30xx) and sm120 (RTX 50xx) - reduces compile time and memory
ENV TORCH_CUDA_ARCH_LIST="8.6 12.0"
# Limit parallel compilation to avoid OOM
ENV MAKEFLAGS=-j8
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV MAX_JOBS=8
ENV NVCC_THREADS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    ninja-build \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install build deps
RUN python3.12 -m pip install --break-system-packages --ignore-installed \
    setuptools wheel setuptools_rust packaging setuptools_scm

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install PyTorch with CUDA 12.8 support
RUN python3.12 -m pip install --break-system-packages --ignore-installed torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copy vllm source
WORKDIR /app
COPY . /app/

# Build vllm with GGUF support - only for sm86 and sm120
RUN python3.12 -m pip install --break-system-packages --ignore-installed -e . --no-build-isolation

# Install gguf Python module
RUN python3.12 -m pip install --break-system-packages gguf

# Default command
CMD ["bash"]
