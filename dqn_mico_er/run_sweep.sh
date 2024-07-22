#!/bin/bash


seeds=(118398 676190 786456 171936 887739 919409 711872 442081 189061 117840)


# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.alpha=0.5 \
        buffer.beta=0.4 \
        logger.group_name=DQN_CartPole_v1_PER_PrioritizedReplay_Alpha0.5_Beta0.4

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.alpha=0.5 \
        buffer.beta=0.5 \
        logger.group_name=DQN_CartPole_v1_PER_PrioritizedReplay_Alpha0.5_Beta0.5

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.alpha=0.5 \
        buffer.beta=0.6 \
        logger.group_name=DQN_CartPole_v1_PER_PrioritizedReplay_Alpha0.5_Beta0.6

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.alpha=0.6 \
        buffer.beta=0.5 \
        logger.group_name=DQN_CartPole_v1_PER_PrioritizedReplay_Alpha0.6_Beta0.5

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.alpha=0.6 \
        buffer.beta=0.6 \
        logger.group_name=DQN_CartPole_v1_PER_PrioritizedReplay_Alpha0.6_Beta0.6

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.alpha=0.8 \
        buffer.beta=0.4 \
        logger.group_name=DQN_CartPole_v1_PER_PrioritizedReplay_Alpha0.8_Beta0.4

    python dqn_cartpole.py -m \
        env.seed=$seed \
        buffer.prioritized_replay=True \
        buffer.alpha=0.8 \
        buffer.beta=0.6 \
        logger.group_name=DQN_CartPole_v1_PER_PrioritizedReplay_Alpha0.8_Beta0.6

    # Execute the script with prioritized_replay=False
    # python dqn_cartpole.py -m \
    #     env.seed=$seed \
    #     buffer.prioritized_replay=False \
    #     logger.group_name=DQN_CartPole_v1_ER_StandardReplay
done