#!/bin/bash

seeds=(118398 676190 786456) #171936 887739)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        env.env_name=CartPole-v1 \
        logger.project_name=dqn_pixels_cart_pole_v1_sweep \
        loss.double_dqn=True \
        logger.cumulative_reward="mean" \
        exp_name=DQN_pixels_baseline_seed_$seed
    
    python dqn.py -m \
        env.seed=$seed \
        env.env_name=CartPole-v1 \
        logger.project_name=dqn_pixels_cart_pole_v1_sweep \
        buffer.prioritized_replay=True \
        loss.double_dqn=True \
        logger.cumulative_reward="mean" \
        exp_name=DQN_pixels_PER_seed_$seed

done