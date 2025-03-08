#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN
#SBATCH --ntasks=1
#SBATCH --time=10:0
#SBATCH --qos=bbshort
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4

# Set W&B API key
export WANDB_API_KEY=<your_wandb_api_key>

set -e
module purge; module load bluebear

apptainer exec torch-rl-gpu.sif python dqn.py -m \
        env.seed=1110 \
        run_name=DQN_atari_1110