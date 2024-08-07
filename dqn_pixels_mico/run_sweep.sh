#!/bin/bash


seeds=(118398) #676190 786456 171936 887739 919409 711872 442081 189061 117840)


# Loop over each seed
for seed in "${seeds[@]}"; do

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     exp_name=DQN_pixels_MICO_num_updates_1

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     buffer.prioritized_replay=True \
    #     exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_4_priority_1

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     buffer.prioritized_replay=True \
    #     buffer.priority_alpha=0.5 \
    #     exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_4_pralpha_0_5

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.priority_type="current_vs_next" \
        exp_name=DQN_pixels_MICO_cnBPER_alpha_0_6_beta_0_4_priority_1

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.priority_type="current_vs_next" \
        buffer.priority_alpha=0.5 \
        exp_name=DQN_pixels_MICO_cnBPER_alpha_0_6_beta_0_4_priority_0_5
done