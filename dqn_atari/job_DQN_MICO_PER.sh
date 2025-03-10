#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN-${GAME_NAME:-Asteroids}
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=18
#SBATCH --qos=bbgpu
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=icelake
#SBATCH --output="outputs/slurm-files/slurm-%A_%a.out"

# Temporary scratch space for I/O efficiency
BB_WORKDIR=$(mktemp -d /scratch/${USER}_${SLURM_JOBID}.XXXXXX)
export TMPDIR=${BB_WORKDIR}

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: No W&B API key provided."
    exit 1
fi

# Set W&B API key from argument and dir
export WANDB_API_KEY=$1
export WANDB_DIR=${BB_WORKDIR}/wandb
mkdir -p $WANDB_DIR

set -e

module purge; module load bluebear
module load bear-apps/2021b
module load Python/3.9.6-GCCcore-11.2.0

export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/my-virtual-env-${BB_CPU}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    python3 -m venv --system-site-packages ${VENV_PATH}
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# Store pip cache in /scratch directory, instead of the default home directory location
PIP_CACHE_DIR="/scratch/${USER}/pip"


seeds=(118398 919409) # 711872) # 442081 189061)

# Loop over each seed and execute tasks sequentially
for seed in "${seeds[@]}"; do
    echo "Starting task with seed $seed at $(date)"
    
    python dqn.py -m \
        env.seed=$seed \
        env.env_name=${GAME_NAME:-Asteroids} \
        loss.mico_loss.enable=True \
        buffer.prioritized_replay.enable=True \
        buffer.prioritized_replay.priority_type=PER \
        run_name=DQN_MICO_${GAME_NAME:-Asteroids}_$seed 

    echo "Completed task with seed $seed at $(date)"

done

# Cleanup
sleep 300  # 5-minute buffer
test -d ${BB_WORKDIR}/wandb/ && /bin/cp -r ${BB_WORKDIR}/wandb/ ./outputs/wandb/
test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}