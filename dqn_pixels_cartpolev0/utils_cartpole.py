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
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="CartPole-v1", frame_skip = 4, 
             device="cpu", seed = 0, is_test=False):

    env = GymEnv(
        env_name,
        frame_skip=frame_skip,
        from_pixels=True,
        pixels_only=False,
        device=device,
    )
    env = TransformedEnv(env)
    # env.append_transform(NoopResetEnv(noops=30, random=True)) # NOTE: Cartpole with no noops will fall into reset in the begining
                                                                # I could use an small noop reset to avoid this, but I think is not necesary
                                                                # in this case. Analyze this later
    if not is_test:
        # env.append_transform(EndOfLifeTransform()) # NOTE: Check my environment is not based on lives (so not important)
        env.append_transform(SignTransform(in_keys=["reward"])) #NOTE: cartpole has no negative rewards
    env.append_transform(ToTensorImage()) 
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500)) # NOTE: Cartpole-v1 has a max of 500 steps
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))
    env.set_seed(seed)

    # NOTE: a rollout will be take a trajectory of frames and group the frames by N=4 sequentially
    # so that the output will be 7x4x84x84 because with a rollout of 10 steps we will have 7 groups of
    # 4 frames each
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


class DQNNetwork(torch.nn.Module):
    """The convolutional network used to compute the agent's Q-values."""
    def __init__(self, 
                 input_shape,
                 num_outputs,
                 num_cells_cnn, 
                 kernel_sizes, 
                 strides, 
                 num_cells_mlp,
                 activation_class,
                 use_batch_norm=False):
        super(DQNNetwork, self).__init__()

        self.activation_class = activation_class()
        self.use_batch_norm = use_batch_norm
      
        # Input shape example: (10, 4, 84, 84)
        _, channels, width, height = input_shape
        self.num_outputs = num_outputs

        # Xavier (Glorot) uniform initialization
        self.initializer = torch.nn.init.xavier_uniform_

        # Convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        in_channels = channels
        for out_channels, kernel_size, stride in zip(num_cells_cnn, kernel_sizes, strides):
            conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            self.conv_layers.append(conv_layer)
            if self.use_batch_norm:
                batch_norm_layer = torch.nn.BatchNorm2d(out_channels)
                self.batch_norm_layers.append(batch_norm_layer)
            in_channels = out_channels

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride  + 1
        
        # Compute the output shape of the conv layers
        width_output = width
        height_output = height
        for kernel_size, stride in zip(kernel_sizes, strides):
            width_output = conv2d_size_out(width_output, kernel_size, stride)
            height_output = conv2d_size_out(height_output, kernel_size, stride)

        cnn_output = width_output * height_output * num_cells_cnn[-1]

        # Fully connected layers
        input_size = cnn_output

        if len(num_cells_mlp) != 0:
            self.fc_layers = torch.nn.ModuleList()
            for units in num_cells_mlp:
                fc_layer = torch.nn.Linear(input_size, units)
                self.fc_layers.append(fc_layer)
                input_size = units
        else:
            self.fc_layers = None
        
        # Final output layer
        self.output_layer = torch.nn.Linear(input_size, self.num_outputs)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.conv_layers:
            self.initializer(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        if self.fc_layers is not None:
            for layer in self.fc_layers:
                self.initializer(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        self.initializer(self.output_layer.weight)
        if self.output_layer.bias is not None:
            torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # x = x.float() / 255.0 # Already normalized by VecNorm
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if self.use_batch_norm:
                x = self.batch_norm_layers[i](x)
            x = self.activation_class(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        if self.fc_layers is not None:
            for fc_layer in self.fc_layers:
                x = self.activation_class(fc_layer(x))
        q_values = self.output_layer(x)
        return q_values


def make_dqn_modules_pixels(proof_environment, policy_cfg):

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
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells= list(policy_cfg.cnn_net.num_cells),
        kernel_sizes=list(policy_cfg.cnn_net.kernel_sizes),
        strides=list(policy_cfg.cnn_net.strides),
    )
    cnn_output = cnn(torch.ones(input_shape))
    mlp = MLP(
        in_features=cnn_output.shape[-1],
        activation_class=activation_class,
        out_features=num_actions,
        num_cells=policy_cfg.mlp_net.num_cells,
    )


    # NOTE: Do I need CompositeSpec here?
    # I think I only need proof_environment.action_spec
    qvalue_module = QValueActor(
        module=torch.nn.Sequential(cnn, mlp),
        spec=CompositeSpec(action=action_spec), 
        in_keys=["pixels"],
    )
    return qvalue_module


def make_dqn_model(env_name, policy_cfg, frame_skip):
    proof_environment = make_env(env_name, frame_skip = frame_skip, device="cpu")
    qvalue_module = make_dqn_modules_pixels(proof_environment, policy_cfg)
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


