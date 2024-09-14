# seeds=(118398 676190 786456 171936 887739) #171936 887739)

# priority_weights=(0.1 0.25 0.5 0.75) #1.0) # Note: 1.0 I just need to calculate 2 seed more
# # priority_weights=(1.0) # Note: 1.0 I just need to calculate 2 seed more

# # Loop over each seed
# for seed in "${seeds[@]}"; do

#     for priority_weight in "${priority_weights[@]}"; do
        
#         python dqn.py -m \
#             env.seed=$seed \
#             buffer.prioritized_replay=True \
#             buffer.mico_priority.enable_mico=True \
#             buffer.mico_priority.priority_type="all_vs_all" \
#             buffer.mico_priority.priority_weight=$priority_weight \
#             buffer.mico_priority.normalize_priorities=False \
#             logger.cumulative_reward="sum" \
#             loss.double_dqn=False \
#             exp_name=BPER_aa_weight_seed_$seed

#     done
# done

# seeds=(118398 676190 786456 171936 887739) #171936 887739)
seeds=(171936 887739) #171936 887739)


priority_weights=(0.1 0.25 0.5 0.75) #1.0) # Note: 1.0 I just need to calculate 2 seed more
# priority_weights=(1.0) # Note: 1.0 I just need to calculate 2 seed more

# Loop over each seed
for seed in "${seeds[@]}"; do

    for priority_weight in "${priority_weights[@]}"; do
        
        python dqn.py -m \
            env.seed=$seed \
            buffer.prioritized_replay=True \
            buffer.mico_priority.enable_mico=True \
            buffer.mico_priority.priority_type="current_vs_next" \
            buffer.mico_priority.priority_weight=$priority_weight \
            buffer.mico_priority.normalize_priorities=False \
            logger.cumulative_reward="sum" \
            loss.double_dqn=False \
            exp_name=BPER_cn_weight_seed_$seed

    done
done