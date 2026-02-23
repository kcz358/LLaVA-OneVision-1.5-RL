ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" \
    PATH="${PATH}:/usr/local/cuda/bin"

RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.12 python3.12-dev python3-pip && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

RUN apt install -y numactl libcairo2

RUN apt update && apt install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

ENV TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0"

WORKDIR /workspace
COPY . /workspace

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Set VIRTUAL_ENV so uv pip install targets the venv created below
ENV VIRTUAL_ENV=/workspace/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN uv pip install -U pip

RUN uv pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y || true

RUN uv pip install torch==2.8.0 torchvision "deepspeed>=0.17.2" pynvml

RUN uv pip install flashinfer-python==0.3.1 --no-build-isolation

RUN cd /workspace/3rdparty/sglang/python && \
    uv pip install ".[all]" && \
    cd /workspace

RUN uv pip install megatron-core==0.13.1 nvidia-ml-py

RUN uv pip install "flash-attn<=2.8.1" --no-build-isolation

RUN cd /workspace/3rdparty/AReaL && \
    uv pip install -e evaluation/latex2sympy && \
    uv pip install "latex2sympy2_extended[antlr4_11_0]" "math-verify[antlr4_11_0]" pysbd polyleven lingua-language-detector && \
    uv pip install -e ".[dev]" --prerelease=allow && \
    cd /workspace

RUN uv pip install openai==2.2.0
RUN uv pip install uvloop==0.21.0
RUN uv pip install cairosvg scikit-image

RUN rm -rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
