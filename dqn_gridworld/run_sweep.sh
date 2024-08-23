#!/bin/bash


seeds=(118398 676190 786456 171936 887739) # 919409 711872 442081 189061 117840)
# seeds=(676190) # 786456 171936 887739 919409 711872 442081 189061 117840)


# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=False \
        buffer.mico_priority.enable_mico=False \
        exp_name=DQN_pixels_baseline_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.mico_priority.enable_mico=False \
        exp_name=DQN_pixels_PER_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=False \
        buffer.mico_priority.enable_mico=True \
        exp_name=DQN_pixels_MICO_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.mico_priority.enable_mico=True \
        buffer.mico_priority.priority_type="all_vs_all" \
        exp_name=DQN_pixels_BPERaa_MICO_seed_$seed

    python dqn.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.mico_priority.enable_mico=True \
        buffer.mico_priority.priority_type="current_vs_next" \
        exp_name=DQN_pixels_BPERcn_MICO_seed_$seed
done