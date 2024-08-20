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

from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    GrayScale,
    CenterCrop,
    GymEnv,
    NoopResetEnv,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
    ObservationNorm,
)
import gymnasium as gym
from torchrl.envs import GymWrapper
import os

from utils_modules import DQNNetwork, MICODQNNetwork

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(cfg_env,  
             device="cpu",
             obs_norm_sd = None):#, is_test=False):
    env_name = cfg_env.env_name
    grid_file = cfg_env.grid_file
    base_name = os.path.basename(grid_file)
    bisimulation_distance_file = os.path.join(
                            os.path.dirname(grid_file),
                            f"distances_{base_name}")

    seed = cfg_env.seed

    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}

    env = gym.make(env_name,
                    render_mode = "rgb_array", 
                    grid_file=grid_file,
                    bisimulation_distance_file=bisimulation_distance_file)
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device=device)
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage()) 
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter()) # NOTE: GridWordv0 has a max of 200 steps
    env.append_transform(DoubleToFloat())
    env.append_transform(ObservationNorm(in_keys=["pixels"], **obs_norm_sd))
    env.set_seed(seed)

    # NOTE: a rollout will be take a trajectory of frames and group the frames by N=4 sequentially
    # so that the output will be 7x4x84x84 because with a rollout of 10 steps we will have 7 groups of
    # 4 frames each
    return env

def get_norm_stats(cfg_env, num_iter = 250):
    # NOTE: num_iter depends of the complexity of the input images (set to 1000 is a good approach)
    test_env = make_env(cfg_env)
    test_env.set_seed(0)
    test_env.transform[-1].init_stats(
        num_iter=num_iter, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    test_env.close()
    del test_env
    return obs_norm_sd


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules_pixels(proof_environment, policy_cfg, enable_mico = False):

    # Define input shape
    input_shape = proof_environment.observation_spec["pixels"].shape
    env_specs = proof_environment.specs

    # NOTE: I think I can change the next two lines by
    # num_outputs = proof_environment.action_spec.shape[-1]
    # action_spec = proof_environment.action_spec.space
    num_actions = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    print(f"Using {policy_cfg.type} for q-net architecture")
    
    # Define Q-Value Module
    activation_class = getattr(torch.nn, policy_cfg.activation)

    if enable_mico:
        q_net = MICODQNNetwork(
            input_shape=input_shape,
            num_outputs=num_actions,
            num_cells_cnn=list(policy_cfg.cnn_net.num_cells),
            kernel_sizes=list(policy_cfg.cnn_net.kernel_sizes),
            strides=list(policy_cfg.cnn_net.strides),
            num_cells_mlp=list(policy_cfg.mlp_net.num_cells),
            activation_class=activation_class,
            use_batch_norm=policy_cfg.use_batch_norm,
        )

        q_net = TensorDictModule(q_net,
            in_keys=["pixels"], 
            out_keys=["action_value", "representation"])

    else:
        q_net = DQNNetwork(
            input_shape=input_shape,
            num_outputs=num_actions,
            num_cells_cnn=list(policy_cfg.cnn_net.num_cells),
            kernel_sizes=list(policy_cfg.cnn_net.kernel_sizes),
            strides=list(policy_cfg.cnn_net.strides),
            num_cells_mlp=list(policy_cfg.mlp_net.num_cells),
            activation_class=activation_class,
            use_batch_norm=policy_cfg.use_batch_norm,
        )

        q_net = TensorDictModule(q_net,
            in_keys=["pixels"], 
            out_keys=["action_value"])


    # NOTE: Do I need CompositeSpec here?
    # I think I only need proof_environment.action_spec
    qvalue_module = QValueActor(
        module=q_net,
        spec=CompositeSpec(action=action_spec), 
        in_keys=["pixels"],
    )
    return qvalue_module


def make_dqn_model(cfg_env, policy_cfg, obs_norm_sd, enable_mico = False):
    proof_environment = make_env(cfg_env, obs_norm_sd = obs_norm_sd, device="cpu")
    qvalue_module = make_dqn_modules_pixels(proof_environment, policy_cfg, enable_mico = enable_mico)   
    del proof_environment
    return qvalue_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
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


