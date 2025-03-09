#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN_MICO
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --qos=bbgpu
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=icelake

# Temporary scratch space for I/O efficiency
BB_WORKDIR=$(mktemp -d /scratch/${USER}_${SLURM_JOBID}.XXXXXX)
export TMPDIR=${BB_WORKDIR}

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: No W&B API key provided."
    exit 1
fi

# Set W&B API key from argument
export WANDB_API_KEY=$1
export TORCH_USE_CUDA_DSA=1  # PyTorch memory handling fix

set -e
module purge; module load bluebear


seeds=(118398 919409 711872) # 442081 189061)

# Loop over each seed and execute tasks sequentially
for seed in "${seeds[@]}"; do
    echo "Starting task with seed $seed at $(date)"
    apptainer exec --nv torch-rl-gpu.sif python dqn.py -m \
        env.seed=$seed \
        loss.mico_loss.enable=True \
        run_name=DQN_MICO_atari_$seed \
        collector.num_iterations=40
    echo "Completed task with seed $seed at $(date)"
done

# Cleanup
test -d ${BB_WORKDIR} && /bin/cp -r ${BB_WORKDIR} .
test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}