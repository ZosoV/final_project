#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN_Asteroids
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --qos=bbdefault
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=36
#SBATCH --nodes=1
#SBATCH --mem=366G
#SBATCH --output="outputs/slurm-files/slurm-DQN-cpu-%A_%a.out"
#SBATCH --constraint=sapphire

GAME_NAME=Asteroids

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: No W&B API key provided."
    exit 1
fi

# Set W&B API key from argument and dir
export WANDB_API_KEY=$1
# export WANDB_DIR=${BB_WORKDIR}/wandb
# mkdir -p $WANDB_DIR

set -x  # Enable debug mode
set -e

module purge; module load bluebear
module load bear-apps/2021b
module load Python/3.9.6-GCCcore-11.2.0

PROJECT_DIR="/rds/projects/g/giacobbm-bisimulation-rl"

export VENV_DIR="${PROJECT_DIR}/virtual-environments"
export VENV_PATH="${VENV_DIR}/cpu_virtual_env-${BB_CPU}"

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


# Perform any required pip installations. For reasons of consistency we would recommend
# that you define the version of the Python module – this will also ensure that if the
# module is already installed in the virtual environment it won't be modified.
pip install torchrl==0.4.0 
pip install tensordict==0.4.0
pip install torch==2.3.1 torchvision==0.18.1
pip install wandb hydra-core tqdm
pip install gymnasium==0.29.1 gymnasium[classic-control]
pip install ale-py gymnasium[other]


seeds=(118398 919409 711872 442081 189061)

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

echo "Starting task with seed $SEED at $(date)"
python dqn_torchrl.py -m \
    device=cpu \
    env.env_name=${GAME_NAME:-Asteroids} \
    env.seed=$SEED \
    run_name=DQN_${GAME_NAME:-Asteroids}_$SEED \
    running_setup.num_envs=4 \
    running_setup.prefetch=8 \
    running_setup.enable_lazy_tensor_buffer=True

echo "Completed task with seed $SEED at $(date)"

echo "Exiting."
exit 0
echo "Exited."
