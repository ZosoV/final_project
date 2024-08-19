#!/bin/bash


# seeds=(118398) # 676190 786456) #171936 887739)

# seeds=(676190 786456) #171936 887739)

# # Loop over each seed
# for seed in "${seeds[@]}"; do

#     python dqn.py -m \
#         env.seed=$seed \
#         env.env_name=LunarLander-v2 \
#         logger.project_name=dqn_pixels_lunarlander_v2_sweep \
#         exp_name=DQN_pixels_baseline_seed_$seed \
    
#     python dqn.py -m \
#         env.seed=$seed \
#         env.env_name=LunarLander-v2 \
#         logger.project_name=dqn_pixels_lunarlander_v2_sweep \
#         buffer.prioritized_replay=True \
#         exp_name=DQN_pixels_PER_seed_$seed \

# done

# seeds=(118398 676190 786456) #171936 887739)
seeds=(786456)

# Loop over each seed
for seed in "${seeds[@]}"; do

    # python dqn.py -m \
    #     env.seed=$seed \
    #     env.env_name=Acrobot-v1 \
    #     logger.project_name=dqn_pixels_acrobot_v1_sweep \
    #     exp_name=DQN_pixels_baseline_seed_$seed \
    
    python dqn.py -m \
        env.seed=$seed \
        env.env_name=Acrobot-v1 \
        logger.project_name=dqn_pixels_acrobot_v1_sweep \
        buffer.prioritized_replay=True \
        exp_name=DQN_pixels_PER_seed_$seed \

done

seeds=(118398 676190 786456) #171936 887739)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        env.env_name=MountainCar-v0 \
        logger.project_name=dqn_pixels_mountain_car_v0_sweep \
        exp_name=DQN_pixels_baseline_seed_$seed \
    
    python dqn.py -m \
        env.seed=$seed \
        env.env_name=MountainCar-v0 \
        logger.project_name=dqn_pixels_mountain_car_v0_sweep \
        buffer.prioritized_replay=True \
        exp_name=DQN_pixels_PER_seed_$seed \

done

seeds=(118398 676190 786456) #171936 887739)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn.py -m \
        env.seed=$seed \
        env.env_name=CarRacing-v2 \
        logger.project_name=dqn_pixels_car_racing_v2_sweep \
        exp_name=DQN_pixels_baseline_seed_$seed \
    
    python dqn.py -m \
        env.seed=$seed \
        env.env_name=CarRacing-v2 \
        logger.project_name=dqn_pixels_car_racing_v2_sweep \
        buffer.prioritized_replay=True \
        exp_name=DQN_pixels_PER_seed_$seed \

done
