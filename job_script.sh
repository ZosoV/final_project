#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --qos=bbshort
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4

set -e
module purge; module load bluebear

# Run a Python script inside the container
apptainer exec torch-rl-gpu.sif python dqn.py
