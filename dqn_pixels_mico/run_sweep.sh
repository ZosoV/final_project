#!/bin/bash

# bash run_sweep_MICO.sh

bash run_sweep_BPER_aa_priority_weight.sh

bash run_sweep_BPER_cn_priority_weight.sh

# seeds=(118398 676190 786456 171936 887739)

# priority_weights=(0.25 0.5 0.75 1.0)

# # Loop over each seed
# for seed in "${seeds[@]}"; do

#     for priority_weight in "${priority_weights[@]}"; do

#         python dqn_cartpole.py -m \
#             env.seed=$seed \
#             env.buffer.prioritized_replay=True \
#             env.buffer.mico_priority.priority_weight=$priority_weight \
#             exp_name=DQN_pixels_MICO_aaBPER_seed_$seed \
        
#     done
    
# done