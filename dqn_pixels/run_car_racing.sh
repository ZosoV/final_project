#!/bin/bash


# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        env.env_name=CarRacing-v2 \
        logger.project_name=dqn_pixels_car_racing_v2_sweep \
        exp_name=DQN_pixels_baseline_seed_$seed
    
    python dqn.py -m \
        env.seed=$seed \
        env.env_name=CarRacing-v2 \
        logger.project_name=dqn_pixels_car_racing_v2_sweep \
        buffer.prioritized_replay=True \
        exp_name=DQN_pixels_PER_seed_$seed

done