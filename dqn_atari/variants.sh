#BPER
python dqn_cpu.py -m \
    env.seed=$SEED \
    env.env_name=${GAME_NAME:-Asteroids} \
    loss.mico_loss.enable=True \
    buffer.prioritized_replay.enable=True \
    buffer.prioritized_replay.priority_type=BPERcn \
    run_name=DQN_MICO_BPER_${GAME_NAME:-Asteroids}_$SEED 

#PER
python dqn.py -m \
    env.seed=$SEED \
    env.env_name=${GAME_NAME:-Asteroids} \
    loss.mico_loss.enable=True \
    buffer.prioritized_replay.enable=True \
    buffer.prioritized_replay.priority_type=PER \
    run_name=DQN_MICO_PER_${GAME_NAME:-Asteroids}_$SEED 

#MICO
python dqn.py -m \
    env.seed=$SEED \
    env.env_name=${GAME_NAME:-Asteroids} \
    loss.mico_loss.enable=True \
    run_name=DQN_MICO_${GAME_NAME:-Asteroids}_$SEED 

#DQN
python dqn_torchrl.py -m \
    env.env_name=${GAME_NAME:-Asteroids} \
    env.seed=$SEED \
    run_name=DQN_${GAME_NAME:-Asteroids}_$SEED \