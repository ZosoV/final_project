#!/bin/bash

seeds=(118398 676190 786456) # 171936 887739)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_mico.py -m \
        env.seed=$seed \
        collector.eps_start=0.1 \
        exp_name=DQN_pixels_MICO_seed_$seed \
    
done