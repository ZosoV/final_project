#!/bin/bash

conda create -y --name final-project python=3.10

conda activate final-project

# Install torch according the nvidia drivers
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install torchrl==0.4.0 torchvision==0.18.1 wandb hydra-core tqdm 
pip install gymnasium==0.29.1 gymnasium[classic-control] 
pip install gymnasium[box2d]

# For gymnasium[box2d]
sudo apt-get update
sudo apt-get install swig
sudo apt-get install build-essential

# ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/zoso/anaconda3/envs/final-project/lib/

pip install swig
pip install gymnasium[box2d]