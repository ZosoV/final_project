# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.nn.functional as F
import torch.optim
from torchrl.data import CompositeSpec
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, MLP, QValueActor
from torchrl.record import VideoRecorder
from tensordict.nn import TensorDictModule
from utils_modules import MICODQNNetwork

from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    RenameTransform,
    RewardClipping
)

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from torchrl.envs import GymWrapper


import numpy as np
np.float_ = np.float64

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="Asteroids", frame_stack = 4, 
             device="cpu", seed = 0, is_test=False):

    full_game_name = f'ALE/{env_name}-v5'


    env = gym.make(full_game_name,
            repeat_action_probability=0.25, # Sticky actions with probability 0.25, as suggested by (Machado et al., 2017).
            frameskip=1,
            max_num_frames_per_episode=100_000,
            )
    # Apply AtariPreprocessing following the Machado et al. (2017) paper
    env = AtariPreprocessing(env,
                            grayscale_newaxis=True)

    env = GymWrapper(env,
                    pixels_only=False, 
                    device=device,
                    categorical_action_encoding=True)
        
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage(in_keys="observation")) 
    env.append_transform(CatFrames(in_keys="observation",N=frame_stack, dim=-3))
    env.append_transform(RenameTransform(["reward"], ["score"]))
    env.append_transform(RewardClipping(clamp_min =-1, clamp_max=1, in_keys=["score"], out_keys=["reward"]))
    env.append_transform(RewardSum(in_keys=["reward"], out_keys=["episode_reward"]))
    env.append_transform(RewardSum(in_keys=["score"], out_keys=["episode_score"]))

    # if not is_test:
    # env.append_transform(EndOfLifeTransform()) # NOTE: Check my environment is not based on lives (so not important)
    #env.append_transform(SignTransform(in_keys=["reward"])) #NOTE: this function simplifies the reward to -1, 0, 1
    # env.append_transform(RewardClipping(clamp_min =-1, clamp_max=1))

    env.append_transform(StepCounter(max_steps=27000)) 
    env.append_transform(DoubleToFloat())
    env.set_seed(seed)
    return env

def update_tensor_dict_next_next_rewards(tensordict):

    def get_next_next_rewards(rewards, dones):
        next_next_rewards = rewards.clone().roll(-1, dims=0)
        next_next_rewards[dones] = 0
        return next_next_rewards

    next_next_rewards = get_next_next_rewards(tensordict['next','reward'], tensordict['next','done'])

    if tensordict.device is not None:
        next_next_rewards = next_next_rewards.to(tensordict.device)

    tensordict.set(
        ("next", "next_reward"),
        next_next_rewards,
        inplace=True,
    )

    # Side effect of the last step. As we cannot get the next_next_reward of the last step
    # we need to remove it or assign it to a value that we know for sure like in cartpole
    # will be always be one, and zero if it is done True.

    # NOTE: We comment this part, because we are using the cartpole environment, but 
    # uncomment and update properly depending the case
    # if not tensordict['next','done'][-1]:
    #     # Two options
    #     # 1. Assign a value
    #     # next_next_rewards[-1] = 1 # Set according the reward of the env 
    #                                # in cartpole will be one so I don't need to do anything
    #                                # in cartpole I don't need even to assign because all are 1
    #     # 2. Remove that last step to avoid problems
    #     data = data[:-1] 

    return tensordict

# ====================================================================
# Model utils
# --------------------------------------------------------------------

def make_dqn_modules_pixels(proof_environment, policy_cfg, enable_mico = False):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs

    # NOTE: I think I can change the next two lines by
    # num_outputs = proof_environment.action_spec.shape[-1]
    # action_spec = proof_environment.action_spec.space
    num_actions = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]
    
    # Define Q-Value Module
    activation_class = getattr(torch.nn, policy_cfg.activation)

    q_net = MICODQNNetwork(
        input_shape=input_shape,
        num_outputs=num_actions,
        num_cells_cnn=list(policy_cfg.cnn_net.num_cells),
        kernel_sizes=list(policy_cfg.cnn_net.kernel_sizes),
        strides=list(policy_cfg.cnn_net.strides),
        num_cells_mlp=list(policy_cfg.mlp_net.num_cells),
        activation_class=activation_class,
        use_batch_norm=policy_cfg.use_batch_norm,
        enable_mico = enable_mico,
    )

    if enable_mico:
        q_net = TensorDictModule(q_net,
            in_keys=["observation"], 
            out_keys=["action_value", "representation"])
    else:
        q_net = TensorDictModule(q_net,
            in_keys=["observation"], 
            out_keys=["action_value"])

    # NOTE: Do I need CompositeSpec here?
    # I think I only need proof_environment.action_spec
    qvalue_module = QValueActor(
        module=q_net,
        spec=CompositeSpec(action=action_spec), 
        in_keys=["observation"],
    )
    return qvalue_module


def make_dqn_model(env_name, policy_cfg, frame_stack, enable_mico = False):
    proof_environment = make_env(env_name, frame_stack = frame_stack, device="cpu")
    qvalue_module = make_dqn_modules_pixels(proof_environment, policy_cfg, enable_mico = enable_mico)
    del proof_environment
    return qvalue_module

# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    eval_seeds = [919409, 711872, 442081, 189061, 117840, 378457, 574025]

    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        test_env.set_seed(eval_seeds[i])
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

def print_hyperparameters(cfg):
    keys = ["env",
            "collector",
            "buffer",
            "policy",
            "optim",
            "loss"]
    
    for key in keys:
        if key in cfg:
            print(f"{key}:")
            for k, v in cfg[key].items():
                print(f"  {k}: {v}")


