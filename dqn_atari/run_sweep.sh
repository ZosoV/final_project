VARIANT=${VARIANT:-DQN}  # Default to DQN if no variant is specified
GAME_NAME=Hero
seeds=(118398 919409 711872 442081 189061)

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

ITERATIONS=25


# python dqn_torchrl.py -m \
#     env.env_name=$GAME_NAME \
#     env.seed=$SEED \
#     run_name=DQN_${GAME_NAME}_$SEED \
#     collector.num_iterations=$ITERATIONS

# echo "Sleeping for 5 minutes..."
# sleep 300  # 5-minute buffer

# Execute based on the selected variant
# python dqn_torchrl.py -m \
#     env.seed=$SEED \
#     env.env_name=$GAME_NAME \
#     loss.mico_loss.enable=True \
#     buffer.prioritized_replay.enable=True \
#     buffer.prioritized_replay.priority_type=BPERcn \
#     run_name=DQN_MICO_BPER_${GAME_NAME}_$SEED \
#     collector.num_iterations=$ITERATIONS

# echo "Sleeping for 5 minutes..."
# sleep 600  # 5-minute buffer


# sleep 300  # 5-minute buffer
CUSTOM_THREADS=8

set -x  # Enable debug mode
set -e  # Stop script on error

# Print variables after and before setting them
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"

# Set the number of threads for the BLAS library
export OMP_NUM_THREADS=$CUSTOM_THREADS
export MKL_NUM_THREADS=$CUSTOM_THREADS
export NUMEXPR_NUM_THREADS=$CUSTOM_THREADS

# Print variables after setting them
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"


# python dqn_torchrl.py -m \
#     env.seed=$SEED \
#     env.env_name=$GAME_NAME \
#     loss.mico_loss.enable=True \
#     run_name=DQN_MICO_${GAME_NAME}_$SEED \
#     collector.num_iterations=$ITERATIONS \
#     running_setup.num_threads=$CUSTOM_THREADS #\
#     # running_setup.device_steps=cuda:0

# echo "Sleeping for 10 minutes..."
# sleep 600  # 10-minute buffer

python dqn_torchrl.py -m \
    env.seed=$SEED \
    env.env_name=$GAME_NAME \
    loss.mico_loss.enable=True \
    buffer.prioritized_replay.enable=True \
    buffer.prioritized_replay.priority_type=PER \
    run_name=DQN_MICO_PER_${GAME_NAME}_$SEED \
    collector.num_iterations=$ITERATIONS \
    running_setup.num_threads=$CUSTOM_THREADS #\
    #running_setup.device_steps=cuda:0

echo "Sleeping for 5 minutes..."