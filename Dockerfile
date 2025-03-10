# Use NVIDIA PyTorch base image with CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

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
