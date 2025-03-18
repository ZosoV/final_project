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
python dqn_torchrl.py -m \
    env.seed=$SEED \
    env.env_name=$GAME_NAME \
    loss.mico_loss.enable=True \
    buffer.prioritized_replay.enable=True \
    buffer.prioritized_replay.priority_type=BPERcn \
    run_name=DQN_MICO_BPER_${GAME_NAME}_$SEED \
    collector.num_iterations=$ITERATIONS

echo "Sleeping for 5 minutes..."
sleep 600  # 5-minute buffer

python dqn_torchrl.py -m \
    env.seed=$SEED \
    env.env_name=$GAME_NAME \
    loss.mico_loss.enable=True \
    buffer.prioritized_replay.enable=True \
    buffer.prioritized_replay.priority_type=PER \
    run_name=DQN_MICO_PER_${GAME_NAME}_$SEED \
    collector.num_iterations=$ITERATIONS

echo "Sleeping for 5 minutes..."
# sleep 300  # 5-minute buffer

# python dqn_torchrl.py -m \
#     env.seed=$SEED \
#     env.env_name=$GAME_NAME \
#     loss.mico_loss.enable=True \
#     run_name=DQN_MICO_${GAME_NAME}_$SEED \
#     collector.num_iterations=$ITERATIONS


