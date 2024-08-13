#!/bin/bash


seeds=(118398 676190 786456 171936 887739)


# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_cartpole.py -m \
        env.seed=$seed \
        exp_name=DQN_pixels_baseline_seed_$seed \
    
    python dqn_pixels.py -m \
        env.seed=$seed \
        env.buffer.prioritized_replay=True \
        exp_name=DQN_pixels_PER_seed_$seed \

done