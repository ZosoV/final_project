#!/bin/bash

seeds=(118398 676190 786456 171936 887739)

priority_weights=(0.25 0.5 0.75 1.0)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_cartpole.py -m \
        env.seed=$seed \
        env.buffer.prioritized_replay=True \
        env.buffer.mico_priority.priority_weight=1.0 \
        exp_name=DQN_pixels_MICO_aaBPER_alpha_beta_0.7_seed_$seed \
    
done