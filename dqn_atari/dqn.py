# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE_Meta file in the root directory of this source tree.

# Modified by Oscar Guarnizo, 2024: 

# Modifications:
# - Added the implementation of the MICO learning
# - Added the implementation of the Prioritized Replay Buffer
# - Added the implementation of the Bisimulation Prioritized Replay Buffer
# - Added control funcitons to log information using weights and biases


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
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from torchrl.record import VideoRecorder
from torchrl.data.replay_buffers.samplers import RandomSampler, PrioritizedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# from torchrl.record.loggers import generate_exp_name, get_logger
from utils_dqn import (
    eval_model,
    make_dqn_model,
    make_env,
    print_hyperparameters,
    update_tensor_dict_next_next_rewards
)
from utils_modules import MICODQNLoss

import tempfile

from collections import deque

import numpy as np
np.float_ = np.float64

@hydra.main(config_path=".", config_name="config", version_base=None)
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

    # Set variables
    frame_skip = cfg.collector.frame_skip
    total_frames = cfg.num_iterations * cfg.collector.training_steps 
    frames_per_batch = cfg.collector.frames_per_batch
    init_random_frames = cfg.collector.init_random_frames
    test_interval = cfg.collector.training_steps


    device = cfg.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Print the current seed and group
    print(f"Running with Seed: {seed} on Device: {device}")

    # Get current date and time
    current_date = datetime.datetime.now()
    date_str = current_date.strftime("%Y_%m_%d-%H_%M_%S")  # Includes date and time

    # Initialize wandb
    wandb.init(
        project=cfg.logger.project_name,
        config=dict(cfg),
        group=cfg.logger.group_name,
        name=f"{cfg.exp_name}_{cfg.env.env_name}_{date_str}",
        mode=cfg.logger.mode,
    )

    # Make the components
    # Policy
    model = make_dqn_model(cfg.env.env_name, 
                           cfg.policy, 
                           frame_skip).to(device)


    # NOTE: annealing_num_steps: number of steps 
    # it will take for epsilon to reach the eps_end value.
    # the decay is linear
    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.epsilon_decay_period,
        eps_init=cfg.collector.epsilon_start,
        eps_end=cfg.collector.epsilon_end,
        spec=model.spec,
    ).to(device)
    model_explore = TensorDictSequential(
        model,
        greedy_module,
    ).to(device)

    # Create the collector
    # NOTE: init_random_frames: Number of frames 
    # for which the policy is ignored before it is called.
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name,
                                frame_skip = frame_skip,
                                device = device, 
                                seed = cfg.env.seed),
        policy=model_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        exploration_type=ExplorationType.RANDOM,
        device=device,
        storing_device=device,
        split_trajs=False,
        # max_frames_per_traj=-1,
        init_random_frames=init_random_frames,
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

    if cfg.buffer.scratch_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        scratch_dir = tempdir.name
    else:
        scratch_dir = cfg.buffer.scratch_dir

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=20,
        storage=LazyMemmapStorage( # NOTE: additional line
            max_size=cfg.buffer.buffer_size,
            scratch_dir=scratch_dir,
        ),
        batch_size=cfg.buffer.batch_size,
        sampler = sampler,
        priority_key="total_priority"
    )
    
    # Create the loss module
    loss_module = MICODQNLoss(
        value_network=model,
        loss_function="l2", 
        delay_value=True, # delay_value=True means we will use a target network
        double_dqn=cfg.loss.double_dqn,
        mico_beta=cfg.loss.mico_loss.mico_beta,
        mico_gamma=cfg.loss.mico_loss.mico_gamma,
        mico_weight=cfg.loss.mico_loss.mico_weight,
        priority_type=cfg.buffer.mico_priority.priority_type,
    )
    # NOTE: additional line for Atari games
    loss_module.set_keys(done="end-of-life", terminated="end-of-life")
    
    loss_module.make_value_estimator(gamma=cfg.loss.gamma) # only to change the gamma value
    loss_module = loss_module.to(device) # NOTE: check if need adding
    
    target_net_updater = HardUpdate(
        loss_module, 
        value_network_update_interval=cfg.loss.target_update_period
    )


    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), 
                                 lr=cfg.optim.lr, #
                                 eps=cfg.optim.eps)
    # Create the logger
    logger = None

    # Create the test environment
    # NOTE: new line
    test_env = make_env(cfg.env.env_name,
                        frame_skip,
                        device,
                        seed=cfg.env.seed, 
                        is_test=True)
    if cfg.logger.video:
        test_env.insert_transform(
            0,
            VideoRecorder(
                logger, tag=f"rendered/{cfg.env.env_name}", in_keys=["pixels"]
            ),
        )
    test_env.eval()

    # Main loop
    collected_frames = 0
    total_episodes = 0
    start_time = time.time()
    grad_clipping = True if cfg.optim.max_grad_norm is not None else False
    max_grad = cfg.optim.max_grad_norm
    batch_size = cfg.buffer.batch_size
    test_interval = test_interval
    num_test_episodes = cfg.logger.num_test_episodes
    prioritized_replay = cfg.buffer.prioritized_replay
    scheduler_activated = cfg.optim.scheduler.active
    frames_per_batch = frames_per_batch
    pbar = tqdm.tqdm(total=total_frames)
    init_random_frames = init_random_frames
    sampling_start = time.time()
    priority_type = cfg.buffer.mico_priority.priority_type
    mico_priority_weight = cfg.buffer.mico_priority.priority_weight
    normalize_priorities = cfg.buffer.mico_priority.normalize_priorities


    # Create a window to store episode rewards as a queue
    cycle_episode_rewards = deque(maxlen=cfg.collector.training_steps)
    cycle_episode_lengths = deque(maxlen=cfg.collector.training_steps)

    # Save the total cumulative rewards over the episodes
    general_cumulative_reward = 0


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

    # NOTE: when analysing this code, take in mind that practically frames
    # are the same as steps
    
    # a data in the for loop will return a batch of transitions but the steps done
    # will corresponds to the size of the batch. Practically, in each transition in the 
    # rollout a frist frame is removed and a new frame is added

    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())

        # NOTE: This reshape must be for frame data (maybe)
        data = data.reshape(-1)
        collected_frames += frames_per_batch
        greedy_module.step(frames_per_batch)

        # Update data before passing to the replay buffer
        # NOTE: It's needed to record the next_next_rewards only
        # for the priority type current vs next

        # NOTE: I need this if I use the distance calculation
        # using the rewards, but as I am using the online network
        # I am not using the rewards

        # if priority_type == "current_vs_next":
        #     data = update_tensor_dict_next_next_rewards(data)

        # NOTE: I need to calculate the mico_distance for the current data
        # to check statistics of the mico_distance
        # data = loss_module.calculate_mico_distance(data)

        replay_buffer.extend(data)

        # Get and log training rewards and episode lengths
        # Collect the episode rewards and lengths in average over the
        # transitions in the current data batch
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]

        # TODO: The basic things I'm gonna log are:
        # num_episodes: number of episodes in the cycle
        # average_return: average return of the episodes in the cycle
        # average_steps_per_second: average steps per second in the cycle


        # When there are at least one done trajectory in the data batch
        if len(episode_rewards) > 0:

            # Logging episodes in a window
            cycle_episode_rewards.extend(episode_rewards.detach().cpu().numpy())

            # Get the number of episodes
            total_episodes += data["next", "done"].sum()

            # Get episode length
            episode_length = data["next", "step_count"][data["next", "done"]]
            cycle_episode_lengths.extend(episode_length.detach().cpu().numpy())


        # Warmup phase (due to the continue statement)
        # Additionally This help us to keep a track of the collected_frames
        # after the init_random_frames
        if collected_frames < init_random_frames:
            continue

        # optimization steps
        training_start = time.time()
        for j in range(num_updates):
            
            sampled_tensordict = replay_buffer.sample(batch_size)
            # TODO: check if the sample is already in the device
            sampled_tensordict = sampled_tensordict.to(device)

            # Also the loss module will use the current and target model to get the q-values
            loss = loss_module(sampled_tensordict)
            q_loss = loss["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            if grad_clipping:
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=max_grad
                )
            optimizer.step()


            # Update the priorities
            if prioritized_replay:
                # norm_td_error = (sampled_tensordict["td_error"] - sampled_tensordict["td_error"].min()) / (sampled_tensordict["td_error"].max() - sampled_tensordict["td_error"].min())
                # norm_mico_distance = (sampled_tensordict["mico_distance"] - sampled_tensordict["mico_distance"].min()) / (sampled_tensordict["mico_distance"].max() - sampled_tensordict["mico_distance"].min())
                norm_td_error = torch.log(sampled_tensordict["td_error"] + 1)
                norm_mico_distance = torch.log(sampled_tensordict["mico_distance"] + 1)
                
                if normalize_priorities:
                    priority = (1 - mico_priority_weight) * norm_td_error + mico_priority_weight * norm_mico_distance
                else:
                    priority = (1 - mico_priority_weight) * sampled_tensordict["td_error"] + mico_priority_weight * sampled_tensordict["mico_distance"]
                replay_buffer.update_priority(index=sampled_tensordict['index'], priority = priority)

            # NOTE: This is only one step (after n-updated steps defined before)
            # the target will update
            target_net_updater.step()
            q_losses[j].copy_(loss["td_loss"].detach())
            mico_losses[j].copy_(loss["mico_loss"].detach())
            total_losses[j].copy_(loss["loss"].detach())

            # Priorities infor to log
            mico_distances[j].copy_(sampled_tensordict["mico_distance"].mean().detach())
            td_errors[j].copy_(sampled_tensordict["td_error"].mean().detach())
            
            if prioritized_replay:
                if normalize_priorities:
                    priorities_per_batch[j].copy_(priority.mean().detach())
                else:
                    priorities_per_batch[j].copy_(torch.log(priority + 1).mean().detach())
                weights_per_batch[j].copy_(sampled_tensordict["_weight"].mean().detach())

            if cfg.logger.save_distributions:
                # NOTE: Have in mind that if you use normalized priorities you are getting
                # the log twice so it's better to get with no normalized priorities
                norm_td_error = (sampled_tensordict["td_error"] - sampled_tensordict["td_error"].mean()) / sampled_tensordict["td_error"].std()
                log_td_error = torch.log(sampled_tensordict["td_error"] + 1)

                norm_mico_distance = (sampled_tensordict["mico_distance"] - sampled_tensordict["mico_distance"].mean()) / sampled_tensordict["mico_distance"].std()
                log_mico_distance = torch.log(sampled_tensordict["mico_distance"] + 1)

                if normalize_priorities:
                    # Max-min normalization of the td_error and mico_distance
                    max_min_norm_td_error = (sampled_tensordict["td_error"] - sampled_tensordict["td_error"].min()) / (sampled_tensordict["td_error"].max() - sampled_tensordict["td_error"].min())
                    max_mim_norm_mico_distance = (sampled_tensordict["mico_distance"] - sampled_tensordict["mico_distance"].min()) / (sampled_tensordict["mico_distance"].max() - sampled_tensordict["mico_distance"].min())
                    priority = (1 - mico_priority_weight) * max_min_norm_td_error + mico_priority_weight * max_mim_norm_mico_distance
                else:
                    priority = (1 - mico_priority_weight) * sampled_tensordict["td_error"] + mico_priority_weight * sampled_tensordict["mico_distance"]

                
                norm_priority = (priority - priority.mean()) / priority.std()
                log_priority = torch.log(priority + 1)

                log_weight = torch.log(sampled_tensordict["_weight"] + 1)
                norm_weight = (sampled_tensordict["_weight"] - sampled_tensordict["_weight"].mean()) / sampled_tensordict["_weight"].std()
                
                log_info.update(
                    {
                        "train/td_error_dist": wandb.Histogram(sampled_tensordict["td_error"].detach().cpu()),
                        "train/log_td_error_dist": wandb.Histogram(log_td_error.detach().cpu()),
                        "train/norm_td_error_dist": wandb.Histogram(norm_td_error.detach().cpu()),
                        "train/mico_distance_dist": wandb.Histogram(sampled_tensordict["mico_distance"].detach().cpu()),
                        "train/log_mico_distance_dist": wandb.Histogram(log_mico_distance.detach().cpu()),
                        "train/norm_mico_distance_dist": wandb.Histogram(norm_mico_distance.detach().cpu()),
                        "train/weight_dist": wandb.Histogram(sampled_tensordict["_weight"].detach().cpu()),
                        "train/log_weight_dist": wandb.Histogram(log_weight.detach().cpu()),
                        "train/norm_weight_dist": wandb.Histogram(norm_weight.detach().cpu()),
                        "train/priority_dist": wandb.Histogram(priority.detach().cpu()),
                        "train/log_priority_dist": wandb.Histogram(log_priority.detach().cpu()),
                        "train/norm_priority_dist": wandb.Histogram(norm_priority.detach().cpu()),
                    }
                )

        training_time = time.time() - training_start


        # Get and log q-values, loss, epsilon, sampling time and training time
        log_info.update(
            {
                "train/q_values": (data["action_value"] * data["action"]).sum().item()
                / frames_per_batch,
                "train/q_mean_values": data["action_value"].mean().item(),
                "train/q_loss": q_losses.mean().item(),
                "train/mico_loss": mico_losses.mean().item(),
                "train/total_loss": total_losses.mean().item(),
                "train/batch_avg_mico_distance": mico_distances.mean().item(),
                "train/batch_avg_td_error": td_errors.mean().item(),
                "train/batch_avg_priority": priorities_per_batch.mean().item(),
                "train/batch_avg_weight": weights_per_batch.mean().item(),
                # "train/buffer_avg_mico_distance": replay_buffer["mico_distance_metadata"].mean().item(),
                "train/epsilon": greedy_module.eps,
                "train/lr": optimizer.param_groups[0]["lr"],
                # "train/sampling_time": sampling_time,
                # "train/training_time": training_time,
            }
        )

        # if cfg.logger.log_buffer_info:
        #     log_info.update(
        #         {
        #             "train/buffer_mico_distance_dist": wandb.Histogram(replay_buffer["mico_distance_metadata"].detach()),
        #         }
        #     )

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
                        # "eval/eval_time": eval_time,
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

    # TODO: Saved the model. Check how to save the model and load
    if cfg.logger.save_model:
        torch.save(model.state_dict(), f"outputs/models/{cfg.exp_name}_{cfg.env.env_name}_{date_str}.pt")


    wandb.finish()


if __name__ == "__main__":
    main()