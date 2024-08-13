seeds=(118398 676190 786456 171936 887739)

priority_weights=(0.25 0.5 0.75 1.0)

# Loop over each seed
for seed in "${seeds[@]}"; do

    for priority_weight in "${priority_weights[@]}"; do
        
        python dqn_cartpole.py -m \
            env.seed=$seed \
            env.buffer.prioritized_replay=True \
            env.buffer.mico_priority.priority_weight=$priority_weight \
            env.buffer.mico_priority.priority_type="current_vs_next" \
            exp_name=DQN_pixels_MICO_cnBPER_seed_$seed \

    done
    
done