# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.optim
from torchrl.data import CompositeSpec
from torchrl.envs import RewardSum, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, QValueActor
from torchrl.record import VideoRecorder
from tensordict.nn import TensorDictModule
import torch.nn.init as init

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="CartPole-v1", device="cpu", seed = 0, from_pixels=False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.set_seed(seed)
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------

class MICOMLPNetwork(torch.nn.Module):
    def __init__(self,
                 in_features,
                 activation_class, 
                 encoder_out_features,
                 q_net_out_features,
                 encoder_num_cells = None,
                 q_net_num_cells = None):
        super(MICOMLPNetwork, self).__init__()

        # TODO: Intitialize the network with 
        # xavier_uniform initialization

        self.activation = activation_class()

        # Encoder section
        if encoder_num_cells is None:
            encoder_num_cells = []
        layers_sizes = [in_features] + encoder_num_cells + [encoder_out_features]

        self.encoder = torch.nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            self.encoder.append(torch.nn.Linear(layers_sizes[i], layers_sizes[i+1]))

        # Q-net section
        if q_net_num_cells is None:
            q_net_num_cells = []

        layers_sizes = [encoder_out_features] + q_net_num_cells + [q_net_out_features.item()]

        self.q_net = torch.nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            self.q_net.append(torch.nn.Linear(layers_sizes[i], layers_sizes[i+1]))

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)       
    
    def forward(self, x):
        for i in range(len(self.encoder) - 1):
            x = self.activation(self.encoder[i](x))

        x = self.encoder[-1](x)
        representation = x
        x = self.activation(x)

        for i in range(len(self.q_net)-1):
            x = self.activation(self.q_net[i](x))

        return self.q_net[-1](x), representation

def make_dqn_modules(proof_environment, policy_cfg):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs

    # NOTE: I think I can change the next two lines by
    # num_outputs = proof_environment.action_spec.shape[-1]
    # action_spec = proof_environment.action_spec.space
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define Q-Value Module and Representations (for MICO)
    if policy_cfg.type == "MLP":
        activation_class = getattr(torch.nn, policy_cfg.activation)
        module = MLP(
            in_features=input_shape[-1],
            activation_class=activation_class,
            out_features=num_outputs,
            num_cells=policy_cfg.network.layers,
        )

    if policy_cfg.type == "MLP_encoder":
        activation_class = getattr(torch.nn, policy_cfg.activation)
        
        module = MICOMLPNetwork(
            in_features=input_shape[-1],
            activation_class=activation_class,
            encoder_num_cells=policy_cfg.encoder.layers,
            encoder_out_features=policy_cfg.encoder.out_features,
            q_net_num_cells=policy_cfg.q_net.layers,
            q_net_out_features=num_outputs,
        )

        module = TensorDictModule(module,
                in_keys=["observation"], 
                out_keys=["action_value", "representation"])

    # NOTE: Do I need CompositeSpec here?
    # I think I only need proof_environment.action_spec
    qvalue_module = QValueActor(
        module=module,
        spec=CompositeSpec(action=action_spec),
        in_keys=["observation"],
    )

    return qvalue_module


def make_dqn_model(env_name, policy_cfg):
    proof_environment = make_env(env_name, device="cpu")
    qvalue_module = make_dqn_modules(proof_environment, policy_cfg)
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
            "optim",
            "loss"]
    
    for key in keys:
        if key in cfg:
            print(f"{key}:")
            for k, v in cfg[key].items():
                print(f"  {k}: {v}")
