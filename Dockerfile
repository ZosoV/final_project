# Use NVIDIA PyTorch base image with CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set up environment variables
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHON_VERSION=3.8.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install the specified Python version
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa -y \
#     && apt-get update \
#     && apt-get install -y python3.8 python3.8-dev python3.8-venv \
#     && rm -rf /var/lib/apt/lists/*

# Set Python 3.8.2 as the default Python version
# RUN apt-get update
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with GPU support and TorchRL
RUN pip install torchrl==0.4.0
RUN pip install tensordict==0.4.0
RUN pip install torch==2.3.1 torchvision==0.18.1
RUN pip install wandb hydra-core tqdm 
RUN pip install gymnasium==0.29.1 gymnasium[classic-control]

# Setting Atari dependencies
RUN pip install ale-py
RUN pip install gymnasium[other]
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]


# docker run --gpus all -it --rm -v /home/zosov/workspace/final_project/dqn_atari:/workspace pytorch-torchrl-gpu python /workspace/dqn.py
