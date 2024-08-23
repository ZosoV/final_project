seeds=(118398 676190 786456) # 171936 887739)

# Loop over each seed
for seed in "${seeds[@]}"; do

    python dqn_mico.py -m \
        env.seed=$seed \
        env.env_name="CartPole-v0" \
        logger.project_name="dqn_pixels_cart_pole_v0_sweep" \
        buffer.prioritized_replay=False \
        logger.cumulative_reward="sum" \
        loss.double_dqn=False \
        exp_name=DQN_pixels_MICO_seed_$seed
    
done