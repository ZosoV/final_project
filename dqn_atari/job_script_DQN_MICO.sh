#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN_MICO-Asteroids
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=14
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
export WANDB_DIR=${BB_WORKDIR}/wandb
mkdir -p $WANDB_DIR

# set -e # Enable 'exit on error'
module purge; module load bluebear


seeds=(118398 919409) # 711872) # 442081 189061)

# Loop over each seed and execute tasks sequentially
for seed in "${seeds[@]}"; do
    echo "Starting task with seed $seed at $(date)"
    apptainer exec --nv torch-rl-gpu.sif python dqn.py -m \
        env.seed=$seed \
        loss.mico_loss.enable=True \
        run_name=DQN_MICO_atari_$seed \
        collector.num_iterations=201
    echo "Completed task with seed $seed at $(date)"

    # Cleanup
    sleep 300  # 5-minute buffer
    test -d ${BB_WORKDIR}/wandb/ && /bin/cp -r ${BB_WORKDIR}/wandb/ ./outputs/wandb/
    test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}

done

# # Cleanup
# sleep 300  # 5-minute buffer
# test -d ${BB_WORKDIR} && /bin/cp -r ${BB_WORKDIR} ./outputs/wandb/
# test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}