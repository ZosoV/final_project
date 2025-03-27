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

# Install Dopmine-RL
RUN pip install dopamine-rl

# Installing Baselines
RUN apt-get update && apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
RUN python3 -m pip install 'tensorflow[and-cuda]'
RUN git clone --single-branch --branch tf2 https://github.com/openai/baselines.git
RUN cd baselines && pip install -e .

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip install ale-py

# Install additional libs
RUN pip install seaborn
RUN pip uninstall -y jax jaxlib
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Jupyter and other dependencies
RUN pip install notebook

# Expose Jupyter's default port
EXPOSE 8888

# Set working directory
WORKDIR /workspace

# Default command
# CMD ["/bin/bash"]
# Run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# docker run --rm -it --gpus all -v "$(pwd)":/workspace dopamine-gpu
