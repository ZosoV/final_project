#!/bin/bash

seeds=(118398 676190 786456) # 171936 887739)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_cartpole.py -m \
        env.seed=$seed \
        exp_name=DQN_pixels_MICO_seed_$seed \
    
done