### Check Similar Conditions as Dopamine Jax DQN Agent

import dopamine.discrete_domains.atari_lib
import dopamine.discrete_domains.run_experiment
import dopamine.jax.agents.dqn.dqn_agent
import dopamine.jax.replay_memory.replay_buffer

- [x] JaxDQNAgent.gamma = 0.99
- [x] JaxDQNAgent.update_horizon = 1 
    - (horizon at which updates are performed, the 'n' in n-step update.)
- [x] JaxDQNAgent.min_replay_history = 20_000  # agent steps
    - number of transitions that should be experienced before the agent begins training its value function.
- [ ] JaxDQNAgent.update_period = 4
- [ ] JaxDQNAgent.target_update_period = 8_000  # agent steps
- [ ] JaxDQNAgent.epsilon_train = 0.01
- [ ] JaxDQNAgent.epsilon_eval = 0.001
- [ ] JaxDQNAgent.epsilon_decay_period = 250_000  # agent steps
# Note: We are using the Adam optimizer by default for JaxDQN, which differs
#       from the original NatureDQN and the dopamine TensorFlow version. In
#       the experiments we have ran, we have found that using Adam yields
#       improved training performance.
- [ ] JaxDQNAgent.optimizer = 'adam'
- [ ] create_optimizer.learning_rate = 6.25e-5
- [ ] create_optimizer.eps = 1.5e-4

atari_lib.create_atari_environment.game_name = 'Pong'
# Sticky actions with probability 0.25, as suggested by (Machado et al., 2017).
- [ ] atari_lib.create_atari_environment.sticky_actions = True
- [ ] create_runner.schedule = 'continuous_train'
- [ ] create_agent.agent_name = 'jax_dqn'
- [ ] create_agent.debug_mode = True
- [ ] Runner.num_iterations = 200
- [ ] Runner.training_steps = 250_000  # agent steps
- [ ] Runner.evaluation_steps = 125_000  # agent steps
- [ ] Runner.max_steps_per_episode = 27_000  # agent steps

- [ ] ReplayBuffer.max_capacity = 1_000_000
- [ ] ReplayBuffer.batch_size = 32