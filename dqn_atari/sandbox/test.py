import gymnasium as gym
import ale_py

# Register the environments
gym.register_envs(ale_py)

from torchrl.envs.libs.gym import GymEnv

env = GymEnv("Pendulum-v1")

print(list(GymEnv.available_envs))
