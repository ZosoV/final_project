#!/bin/bash

seeds=(118398) # 919409 711872 442081 189061 117840)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        run_name=DQN_atari_$seed
    
    python dqn.py -m \
        env.seed=$seed \
        loss.mico_loss.enable=True \
        run_name=DQN_MICO_atari_$seed

done