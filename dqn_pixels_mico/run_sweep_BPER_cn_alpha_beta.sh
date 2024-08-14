seeds=(118398 676190 786456) #171936 887739)

alphas=(0.6 0.7 0.8)
betas=(0.4 0.6 0.8)


# Loop over each seed
for seed in "${seeds[@]}"; do

    for alpha in "${alphas[@]}"; do

        for beta in "${betas[@]}"; do

            python dqn_cartpole.py -m \
                env.seed=$seed \
                buffer.prioritized_replay=True \
                buffer.alpha=$alpha \
                buffer.beta=$beta \
                buffer.mico_priority.priority_type="current_vs_next" \
                buffer.mico_priority.priority_weight=1.0 \
                exp_name=DQN_pixels_MICO_BPER_cn_alpha_beta_seed_$seed \

        done

    done
done