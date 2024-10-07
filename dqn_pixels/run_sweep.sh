#!/bin/bash

seeds=(118398 676190 786456 171936 887739) # 919409 711872 442081 189061 117840)
# seeds=(171936 887739) # 919409 711872 442081 189061 117840)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        env.env_name=LunarLander-v2 \
        logger.project_name=dqn_pixels_lunar_lander_v2_sweep_3 \
        exp_name=DQN_pixels_baseline_seed_$seed
    
    # python dqn.py -m \
    #     env.seed=$seed \
    #     env.env_name=LunarLander-v2 \
    #     logger.project_name=dqn_pixels_lunar_lander_v2_sweep_2 \
    #     buffer.prioritized_replay=True \
    #     exp_name=DQN_pixels_PER_seed_$seed

done
