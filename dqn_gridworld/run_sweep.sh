#!/bin/bash


seeds=(118398) #676190 786456 171936 887739 919409 711872 442081 189061 117840)


# Loop over each seed
for seed in "${seeds[@]}"; do

    # python dqn.py -m \
    #     env.seed=$seed \
    #     exp_name=DQN_pixels_baseline_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        exp_name=DQN_pixels_PER_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        env.enable_mico=True \
        exp_name=DQN_pixels_MICO_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        env.enable_mico=True \
        exp_name=DQN_pixels_BPERaa_MICO_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        env.enable_mico=True \
        buffer.mico_priority.priority_type="current_vs_next" \
        exp_name=DQN_pixels_BPERcn_MICO_seed_$seed
done