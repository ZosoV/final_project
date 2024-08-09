#!/bin/bash


seeds=(118398) #676190 786456 171936 887739 919409 711872 442081 189061 117840)


# Loop over each seed
for seed in "${seeds[@]}"; do

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     exp_name=DQN_pixels_baseline

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     buffer.prioritized_replay=True \
    #     exp_name=DQN_pixels_PER_alpha_0_6_beta_0_4

    python dqn_cartpole.py -m \
        env.seed=$seed \
        optim.scheduler.active=True \
        exp_name=DQN_pixels_baseline_scheduler_True

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        optim.scheduler.active=True \
        exp_name=DQN_pixels_PER_alpha_0_6_beta_0_4_scheduler_True
done