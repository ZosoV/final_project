#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN
#SBATCH --ntasks=1
#SBATCH --time=6-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbgpu
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a30:1

# Set W&B API key
export WANDB_API_KEY=<your_wandb_api_key>

set -e
module purge; module load bluebear


seeds=(118398 919409 711872 442081 189061)

# Loop over each seed and execute tasks sequentially
for seed in "${seeds[@]}"; do
    echo "Starting task with seed $seed at $(date)"
    apptainer exec torch-rl-gpu.sif python dqn.py -m \
        env.seed=$seed \
        run_name=DQN_atari_$seed \
        collector.num_iterations=40 \
    echo "Completed task with seed $seed at $(date)"
done
