#!/bin/bash


seeds=(118398) #676190 786456 171936 887739 919409 711872 442081 189061 117840)


# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_cartpole.py -m \
        env.seed=$seed \
        collector.annealing_frames=500000 \
        exp_name=DQN_pixels_annealing_frames_500_000

    python dqn_cartpole.py -m \
        env.seed=$seed \
        collector.annealing_frames=1000000 \
        exp_name=DQN_pixels_annealing_frames_1_000_000

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.batch_size=128 \
        exp_name=DQN_pixels_batch_size_128
    
    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.buffer_size=200000 \
        exp_name=DQN_pixels_buffer_size_200_000
    
    python dqncartpole.py -m \
        env.seed=$seed \
        optim.lr=0.0001 \
        exp_name=DQN_pixels_lr_0_0001

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     optim.lr=0.0005 \
    #     exp_name=DQN_pixels_lr_0_0005

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     optim.lr=0.001 \
    #     exp_name=DQN_pixels_lr_0_001

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     optim.lr=0.005 \
    #     exp_name=DQN_pixels_lr_0_0005

    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     loss.hard_update_freq: 5000 \
    #     exp_name=DQN_pixels_hard_update_freq_5000

done