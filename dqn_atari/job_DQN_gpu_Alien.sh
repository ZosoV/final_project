#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN-Alien
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --qos=bbgpu
#SBATCH --cpus-per-task=14
##SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a30:1
#SBATCH --output="outputs/slurm-files/slurm-DQN-%A_%a.out"

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load tqdm/4.66.1-GCCcore-12.3.0
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
# module load bear-apps/2022a
# module load wandb/0.13.6-GCC-11.3.0

GAME_NAME=Alien
VARIANT=${VARIANT:-DQN}  # Default to DQN if no variant is specified
CUSTOM_THREADS=14
ITERATIONS=40

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



# pip install torch==2.3.1 torchvision==0.18.1

PROJECT_DIR="/rds/projects/g/giacobbm-bisimulation-rl"
export VENV_DIR="${PROJECT_DIR}/virtual-environments"
export VENV_PATH="${VENV_DIR}/gpu-virtual-env-${BB_CPU}"

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
pip install hydra-core wandb
pip install gymnasium==0.29.1 gymnasium[classic-control]
pip install ale-py gymnasium[other]

seeds=(118398 919409 711872 442081 189061)

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

echo "Starting task with seed $SEED at $(date)"

# Print current OMP_NUM_THREADS and MKL_NUM_THREADS
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

# Set the number of threads for MKL and OMP
export OMP_NUM_THREADS=$CUSTOM_THREADS
export MKL_NUM_THREADS=$CUSTOM_THREADS

# Execute based on the selected variant
if [ "$VARIANT" == "BPER" ]; then
    python dqn_torchrl.py -m \
        env.seed=$SEED \
        env.env_name=$GAME_NAME \
        collector.num_iterations=$ITERATIONS \
        loss.mico_loss.enable=True \
        buffer.prioritized_replay.enable=True \
        buffer.prioritized_replay.priority_type=BPERcn \
        run_name=DQN_MICO_BPER_${GAME_NAME}_${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
        running_setup.num_threads=$CUSTOM_THREADS

    # wandb sync outputs/DQN_MICO_BPER_${GAME_NAME}_$SEED

elif [ "$VARIANT" == "PER" ]; then
    python dqn_torchrl.py -m \
        env.seed=$SEED \
        env.env_name=$GAME_NAME \
        collector.num_iterations=$ITERATIONS \
        loss.mico_loss.enable=True \
        buffer.prioritized_replay.enable=True \
        buffer.prioritized_replay.priority_type=PER \
        run_name=DQN_MICO_PER_${GAME_NAME}_${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
        running_setup.num_threads=$CUSTOM_THREADS

    # wandb sync outputs/DQN_MICO_PER_${GAME_NAME}_$SEED

elif [ "$VARIANT" == "MICO" ]; then
    python dqn_torchrl.py -m \
        env.seed=$SEED \
        env.env_name=$GAME_NAME \
        collector.num_iterations=$ITERATIONS \
        loss.mico_loss.enable=True \
        run_name=DQN_MICO_${GAME_NAME}_${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
        running_setup.num_threads=$CUSTOM_THREADS
    
    # wandb sync outputs/DQN_MICO_${GAME_NAME}_$SEED

elif [ "$VARIANT" == "DQN" ]; then
    python dqn_torchrl.py -m \
        env.env_name=$GAME_NAME \
        env.seed=$SEED \
        collector.num_iterations=$ITERATIONS \
        run_name=DQN_${GAME_NAME}_${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
        running_setup.num_threads=$CUSTOM_THREADS \
        running_setup.prefetch=14 \
        running_setup.num_envs=4 \
        collector.frames_per_batch=4 \
        loss.num_updates=1        

    # wandb sync outputs/DQN_${GAME_NAME}_$SEED

else
    echo "Unknown variant: $VARIANT"
    exit 1
fi



echo "Completed task with seed $SEED at $(date)"


# Cleanup
# sleep 300  # 5-minute buffer
test -d ${BB_WORKDIR}/wandb/ && /bin/cp -r ${BB_WORKDIR}/wandb/ ./outputs/wandb/
test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}

echo "Exiting."
exit 0
echo "Exited."
