#!/bin/bash

conda create -y --name final-project python=3.10

conda activate final-project

# Install torch according the nvidia drivers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install torchrl wandb hydra-core tqdm gymnasium gymnasium[classic-control] gymnasium[box2d]