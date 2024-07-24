# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import datetime

import hydra
import torch.nn
import torch.optim
import tqdm
import wandb

import random
import numpy as np
import torch

from tensordict.nn import TensorDictSequential
# from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record import VideoRecorder
from torchrl.data.replay_buffers.samplers import RandomSampler, PrioritizedSampler

# from torchrl.record.loggers import generate_exp_name, get_logger
from utils_cartpole import eval_model, make_dqn_model, make_env, print_hyperparameters

# TODO: TODOS:
# [ ] Check the parameters of the replay buffer what is happening in the iteration 210000
# [ ] Check if seed is working correctly with an small example
# [ ] Set a variable of experiment name outside
# [ ] Send everything to the gpu only for this example (I could have faster executions)
# I am only using 143MB from 8GB of GPU memory
# [ ] Check how smoothing works in wandb and check if there is another way to calculate that expected reward
# For the implementation of the replay review this tutorial
# things about the collector and the replay buffer
# https://pytorch.org/rl/stable/tutorials/coding_ddpg.html#coding-ddpg

@hydra.main(config_path=".", config_name="config_cartpole", version_base=None)
def main(cfg: "DictConfig"):

    # Set seeds for reproducibility
    seed = cfg.env.seed
    random.seed(seed)
    np.random.seed(seed)
    # TODO: maybe better to set before the loop???
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # NOTE: This step is needed to have reproducibility
        # But it reduces a little bit the performance
        # if I don't need reproducibility I could comment this line
        torch.backends.cudnn.benchmark = False

    # Print the current seed and group
    print(f"Running with Seed: {seed}")
    print(f"Group: {cfg.logger.group_name}")

    device = cfg.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Get current date and time
    current_date = datetime.datetime.now()
    date_str = current_date.strftime("%Y_%m_%d-%H_%M_%S")  # Includes date and time

    # Initialize wandb
    wandb.init(
        project=cfg.logger.project_name,
        config=dict(cfg),
        group=cfg.logger.group_name,
        name=f"{cfg.exp_name}_{cfg.env.env_name}_{date_str}"
    )

    # Make the components
    # Policy
    model = make_dqn_model(cfg.env.env_name, cfg.policy)


    # NOTE: annealing_num_steps: number of steps 
    # it will take for epsilon to reach the eps_end value.
    # the decay is linear (slower than exponential)
    # max(
    #     self.eps_end.item(),
    #     (
    #         self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
    #     ).item(),
    # )
    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=model.spec,
    )
    model_explore = TensorDictSequential(
        model,
        greedy_module,
    ).to(device)

    # Create the collector
    # NOTE: init_random_frames: Number of frames 
    # for which the policy is ignored before it is called.
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, "cpu", cfg.env.seed),
        policy=model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device="cpu",
        storing_device="cpu",
        max_frames_per_traj=-1,
        init_random_frames=cfg.collector.init_random_frames,
    )

    # Create the replay buffer
    if cfg.buffer.prioritized_replay:
        print("Using Prioritized Replay Buffer")
        sampler = PrioritizedSampler(
            max_capacity=cfg.buffer.buffer_size, 
            alpha=cfg.buffer.alpha, 
            beta=cfg.buffer.beta)
    else:
        sampler = RandomSampler()
        
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=10,
        storage=LazyTensorStorage(
            max_size=cfg.buffer.buffer_size,
            device="cpu",
        ),
        batch_size=cfg.buffer.batch_size,
        sampler = sampler
    )
    
    # Create the loss module
    loss_module = DQNLoss(
        value_network=model,
        loss_function="l2", 
        delay_value=True, # delay_value=True means we will use a target network
    )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma) # only to change the gamma value
    loss_module = loss_module.to(device)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    logger = None
    # if cfg.logger.backend:
    #     exp_name = generate_exp_name("DQN", f"CartPole_{cfg.env.env_name}")
    #     logger = get_logger(
    #         cfg.logger.backend,
    #         logger_name="dqn",
    #         experiment_name=exp_name,
    #         wandb_kwargs={
    #             "config": dict(cfg),
    #             "project": cfg.logger.project_name,
    #             "group": cfg.logger.group_name,
    #         },
    #     )

    # Create the test environment
    test_env = make_env(cfg.env.env_name, "cpu", from_pixels=cfg.logger.video, seed=cfg.env.seed)
    if cfg.logger.video:
        test_env.insert_transform(
            0,
            VideoRecorder(
                logger, tag=f"rendered/{cfg.env.env_name}", in_keys=["pixels"]
            ),
        )

    # Main loop
    collected_frames = 0
    total_episodes = 0
    start_time = time.time()
    num_updates = cfg.loss.num_updates
    batch_size = cfg.buffer.batch_size
    test_interval = cfg.logger.test_interval
    num_test_episodes = cfg.logger.num_test_episodes
    frames_per_batch = cfg.collector.frames_per_batch
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    init_random_frames = cfg.collector.init_random_frames
    sampling_start = time.time()
    q_losses = torch.zeros(num_updates, device=device)


    # NOTE: IMPORTANT: collectors allows me to collect transitions in a different way
    # than the one I am get used to.
    # Practically, with the collector, I defined before hand all the number of interactions 
    # that I want to have.

    # For example, if I have a total of 500100 frames with batches of 10
    # I gonna have interaction of 10 frames for 50010 times.
    # and in each interaction, I will do somethings after the warmup phase

    # However, the metrics are not calculate per interaction, but an average of the
    # metrics of the transitions in the current data batch

    # The policy acts internatly in the collector, so I don't need to worry about 
    # the step of the policy, the collector will do it for me.

    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())

        # NOTE: This reshape must be for frame data (maybe)
        data = data.reshape(-1)
        current_frames = data.numel()
        replay_buffer.extend(data)
        collected_frames += current_frames
        greedy_module.step(current_frames)

        # Get the number of episodes
        total_episodes += data["next", "done"].sum()

        # Get and log training rewards and episode lengths
        # Collect the episode rewards and lengths in average over the
        # transitions in the current data batch
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]

        # When there are at least one done trajectory in the data batch
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_length_mean = episode_length.sum().item() / len(episode_length)

            # NOTE: this log will be updated only if there is a new episode in the current
            # data batch gotten from interaction with the environment
            log_info.update(
                {
                    "train/episode_reward": episode_reward_mean,
                    # "train/episode_length": episode_length_mean,
                }
            )

        # Warmup phase (due to the continue statement)
        # Additionally This help us to keep a track of the collected_frames
        # after the init_random_frames
        if collected_frames < init_random_frames:
            wandb.log(log_info, step=collected_frames)
            continue

        # optimization steps
        training_start = time.time()
        for j in range(num_updates):
            sampled_tensordict = replay_buffer.sample(batch_size)
            # TODO: check if the sample is already in the device
            sampled_tensordict = sampled_tensordict.to(device)

            # Also the loss module will use the current and target model to get the q-values
            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()

            # Update the priorities
            if cfg.buffer.prioritized_replay:
                replay_buffer.update_priority(index=sampled_tensordict['index'], priority = sampled_tensordict['td_error'])

            # NOTE: This is only one step (after n-updated steps defined before)
            # the target will update
            target_net_updater.step()
            q_losses[j].copy_(q_loss.detach())
        training_time = time.time() - training_start

        # Get and log q-values, loss, epsilon, sampling time and training time
        log_info.update(
            {
                "train/q_values": (data["action_value"] * data["action"]).sum().item()
                / frames_per_batch,
                "train/q_loss": q_losses.mean().item(),
                "train/epsilon": greedy_module.eps,
                "train/episode_per_chunk": data["next", "done"].sum().item(),
                # "train/sampling_time": sampling_time,
                # "train/training_time": training_time,
            }
        )

        # Get and log evaluation rewards and eval time
        # NOTE: As I'm using only the model and not the model_explore that will deterministic I think
        with torch.no_grad(): #, set_exploration_type(ExplorationType.DETERMINISTIC):

            # NOTE: Check how we are using the frames here because it seems that I am dividing 
            # 10 for 50000
            prev_test_frame = ((i - 1) * frames_per_batch) // test_interval
            cur_test_frame = (i * frames_per_batch) // test_interval
            final = current_frames >= collector.total_frames

            # compara prev_test_frame < cur_test_frame is the same as current_frames % test_interval == 0
            if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
                model.eval()
                eval_start = time.time()
                test_rewards = eval_model(model, test_env, num_test_episodes)
                eval_time = time.time() - eval_start
                model.train()
                log_info.update(
                    {
                        "eval/reward": test_rewards,
                        "eval/eval_time": eval_time,
                    }
                )

        # Log all the information
        wandb.log(log_info, step=collected_frames)

        # update weights of the inference policy
        # NOTE: Updates the policy weights if the policy of the data 
        # collector and the trained policy live on different devices.
        collector.update_policy_weights_()
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(execution_time)))
    print(f"Collected Frames: {collected_frames}, Total Episodes: {total_episodes}")
    print(f"Training took {formatted_time} (HH:MM:SS) to finish")
    print("Hyperparameters used:")
    print_hyperparameters(cfg)

    # TODO: Print the hyperparameters used
    # wandb.finish()


if __name__ == "__main__":
    main()
