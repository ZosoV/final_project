#!/bin/bash

seeds=(786456 171936 887739) # 919409 711872 442081 189061 117840)

priority_weights=(0.0 1.0) # Note: 1.0 I just need to calculate 2 seed more
# priority_weights=(1.0) # Note: 1.0 I just need to calculate 2 seed more

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_mico.py -m \
        env.seed=$seed \
        env.env_name="LunarLander-v2" \
        logger.project_name="dqn_pixels_lunar_lander_v2_sweep_2" \
        exp_name=DQN_MICO_seed_$seed
    
done

seeds=(118398 676190 786456 171936 887739) # 919409 711872 442081 189061 117840)

priority_weights=(0.0 1.0) # Note: 1.0 I just need to calculate 2 seed more
# priority_weights=(1.0) # Note: 1.0 I just need to calculate 2 seed more

# Loop over each seed
for seed in "${seeds[@]}"; do

    for priority_weight in "${priority_weights[@]}"; do
        
        python dqn_mico.py -m \
            env.seed=$seed \
            env.env_name="LunarLander-v2" \
            logger.project_name="dqn_pixels_lunar_lander_v2_sweep_2" \
            buffer.prioritized_replay=True \
            buffer.mico_priority.priority_type="all_vs_all" \
            buffer.mico_priority.priority_weight=$priority_weight \
            buffer.mico_priority.normalize_priorities=False \
            exp_name=DQN_MICO_BPER_aa_seed_$seed

    done
    
done

# seeds=(118398 676190 786456) #171936 887739)

# # seeds=(118398) #171936 887739)


# priority_weights=(0.1) #1.0) # Note: 1.0 I just need to calculate 2 seed more
# # priority_weights=(1.0) # Note: 1.0 I just need to calculate 2 seed more

# Loop over each seed
for seed in "${seeds[@]}"; do

    for priority_weight in "${priority_weights[@]}"; do
        
        python dqn_mico.py -m \
            env.seed=$seed \
            env.env_name="LunarLander-v2" \
            logger.project_name="dqn_pixels_lunar_lander_v2_sweep_2" \
            buffer.prioritized_replay=True \
            buffer.mico_priority.priority_type="current_vs_next" \
            buffer.mico_priority.priority_weight=$priority_weight \
            buffer.mico_priority.normalize_priorities=False \
            exp_name=DQN_MICO_BPER_cn_seed_$seed

    done
    
done