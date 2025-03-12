#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN-${GAME_NAME:-Asteroids}
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --qos=bbgpu
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a100:1
#SBATCH --output="outputs/slurm-files/slurm-DQN-%A_%a.out"
#SBATCH --constraint=icelake

# Temporary scratch space for I/O efficiency
BB_WORKDIR=$(mktemp -d /scratch/${USER}_${SLURM_JOBID}.XXXXXX)
# BB_WORKDIR=$(mktemp -d /rds/projects/g/giacobbm-bisimulation-rl/${USER}_${SLURM_JOBID}.XXXXXX)
export TMPDIR=${BB_WORKDIR}
# export EXP_BUFF=${BB_WORKDIR}

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
python dqn.py -m \
    env.env_name=${GAME_NAME:-Asteroids} \
    env.seed=$SEED \
    run_name=DQN_${GAME_NAME:-Asteroids}_$SEED
echo "Completed task with seed $SEED at $(date)"


# Cleanup
sleep 300  # 5-minute buffer
# test -d ${BB_WORKDIR}/wandb/ && /bin/cp -r ${BB_WORKDIR}/wandb/ ./outputs/wandb/
test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}

echo "Exiting."
exit 0
echo "Exited."