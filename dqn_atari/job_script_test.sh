#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN-Asteroids
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=14
#SBATCH --qos=bbgpu
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=icelake

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: No W&B API key provided."
    exit 1
fi

# Set W&B API key from argument and dir
export WANDB_API_KEY=$1

set -e # Enable 'exit on error'
module purge; module load bluebear

apptainer exec --nv torch-rl-gpu.sif python dqn.py

sleep 300  # 5-minute buffer
