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
    #     buffer.alpha=0.6 \
    #     buffer.beta=0.4 \
    #     optim.lr=0.0005 \
    #     exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_4_pralpha_0_5_lr_0_0005

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     buffer.prioritized_replay=True \
    #     buffer.priority_alpha=0.5 \
    #     buffer.alpha=0.6 \
    #     buffer.beta=0.4 \
    #     optim.lr=0.0001 \
    #     exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_4_pralpha_0_5_lr_0_0001

    # NOTE: In theory, as the q_loss becomes more stable we could play with the learning rate
    # It seems that the learning rate need a high value in the beginning a low value in the end
    # so we could use a scheduler to decrease the learning rate
    # As the learning is more stable we can do that.

    # TODO: Try this two in this computer and try the original for more time steps
    # TODO: Try to normalize the distance
    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     buffer.prioritized_replay=True \
    #     buffer.mico_priority.priority_weight=0.5 \
    #     buffer.alpha=0.6 \
    #     buffer.beta=0.4 \
    #     exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_4_prweight_0_5

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     buffer.prioritized_replay=True \
    #     buffer.mico_priority.priority_weight=0.5 \
    #     buffer.mico_priority.normalize_priorities=True \
    #     buffer.alpha=0.6 \
    #     buffer.beta=0.4 \
    #     exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_4_prweight_0_5_norm_priorities_True

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.mico_priority.priority_weight=0.5 \
        buffer.alpha=0.6 \
        buffer.beta=0.4 \
        optim.scheduler.active=True \
        exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_4_prweight_0_5_scheduler_True


    # In theory, I could increase the beta to account for the bias because of the same MICO loss
    # will encourage exploration later on [Check this. I'm not sure if this is true]
    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.mico_priority.priority_weight=0.5 \
        buffer.alpha=0.6 \
        buffer.beta=0.6 \
        exp_name=DQN_pixels_MICO_aaBPER_alpha_0_6_beta_0_6_prweight_0_5

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     exp_name=DQN_pixels_MICO
    
done