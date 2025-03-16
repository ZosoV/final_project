#!/bin/bash

seeds=(118398) # 919409 711872 442081 189061 117840)

GAME_NAME=Asteroids

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        env.env_name=${GAME_NAME:-Asteroids} \
        loss.mico_loss.enable=True \
        buffer.prioritized_replay.enable=True \
        buffer.prioritized_replay.priority_type=PER \
        run_name=DQN_MICO_PER_${GAME_NAME:-Asteroids}_$seed 

done