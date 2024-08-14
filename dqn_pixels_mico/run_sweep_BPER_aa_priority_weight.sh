seeds=(118398 676190 786456) # 171936 887739)

priority_weights=(0.25 0.5 0.75) #1.0) # Note: 1.0 I just need to calculate 2 seed more

# Loop over each seed
for seed in "${seeds[@]}"; do

    for priority_weight in "${priority_weights[@]}"; do
        
        python dqn_cartpole.py -m \
            env.seed=$seed \
            buffer.prioritized_replay=True \
            buffer.alpha=? \ # TBD
            buffer.beta=? \ # TBD
            buffer.mico_priority.priority_type="all_vs_all" \
            buffer.mico_priority.priority_weight=$priority_weight \
            buffer.mico_priority.normalize_priorities=False \
            exp_name=DQN_pixels_MICO_BPER_aa_priority_weight_seed_$seed \

    done
    
done