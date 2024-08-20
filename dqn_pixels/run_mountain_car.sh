#!/bin/bash

seeds=(676190 786456) #171936 887739)


python dqn.py -m \
    env.seed=118398 \
    env.env_name=MountainCar-v0 \
    logger.project_name=dqn_pixels_mountain_car_v0_sweep \
    buffer.prioritized_replay=True \
    exp_name=DQN_pixels_PER_seed_118398

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        env.env_name=MountainCar-v0 \
        logger.project_name=dqn_pixels_mountain_car_v0_sweep \
        exp_name=DQN_pixels_baseline_seed_$seed
    
    python dqn.py -m \
        env.seed=$seed \
        env.env_name=MountainCar-v0 \
        logger.project_name=dqn_pixels_mountain_car_v0_sweep \
        buffer.prioritized_replay=True \
        exp_name=DQN_pixels_PER_seed_$seed

done