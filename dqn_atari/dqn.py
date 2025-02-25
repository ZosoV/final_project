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
from utils_modules import MICODQNLoss, MovingAverageNormalization, DQNLogger

import tempfile

from collections import deque

import numpy as np
np.float_ = np.float64

import gymnasium as gym
import ale_py


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: "DictConfig"):

    # Register the environments
    gym.register_envs(ale_py)

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
        # torch.backends.cudnn.benchmark = False

    # Set variables
    frame_skip = cfg.collector.frame_skip
    total_frames = cfg.collector.num_iterations * cfg.collector.training_steps 
    frames_per_batch = cfg.collector.frames_per_batch
    init_random_frames = cfg.collector.init_random_frames
    training_steps = cfg.collector.training_steps

    grad_clipping = True if cfg.optim.max_grad_norm is not None else False
    max_grad = cfg.optim.max_grad_norm
    batch_size = cfg.buffer.batch_size
    prioritized_replay = cfg.buffer.prioritized_replay
    mico_priority_weight = cfg.buffer.mico_priority.priority_weight
    enable_mico = cfg.loss.mico_loss.enable

    # Evaualation
    enable_evaluation = cfg.logger.evaluation.enable
    num_eval_episodes = cfg.logger.evaluation.num_episodes


    device = cfg.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Print the current seed and group
    print(f"Running with Seed: {seed} on Device: {device}")

    # Init logger
    current_date = datetime.datetime.now()
    date_str = current_date.strftime("%Y_%m_%d-%H_%M_%S")  # Includes date and time
    logger = DQNLogger(
        project_name=cfg.logger.project_name,
        run_name=f"{cfg.exp_name}_{cfg.env.env_name}_{date_str}",
        mode=cfg.logger.mode,
        config=dict(cfg),
        log_interval=cfg.collector.training_steps,
    )

    # Make the components
    # Policy
    model = make_dqn_model(cfg.env.env_name, 
                           cfg.policy, 
                           frame_skip,
                           enable_mico).to(device)


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

    # Create a moving average for the priorities
    if cfg.buffer.mico_priority.moving_average is not None:
        normalizer = MovingAverageNormalization(momentum=0.1)
    
    # Create the loss module
    if enable_mico:
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
    else:
        loss_module = DQNLoss(
            value_network=model,
            loss_function="l2", 
            delay_value=True, # delay_value=True means we will use a target network
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


    # Create the test environment
    # NOTE: new line
    test_env = make_env(cfg.env.env_name,
                        frame_skip,
                        device,
                        seed=cfg.env.seed, 
                        is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0 # Also corresponds to the current step
    total_episodes = 0
    pbar = tqdm.tqdm(total=total_frames)

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

    start_time = time.time()

    c_iter = iter(collector)

    for iteration in range(cfg.num_iterations):

        i = 0
        
        training_start = time.time()
        while i < training_steps:
            data = next(c_iter)
        
            pbar.update(data.numel())

            # NOTE: This reshape must be for frame data (maybe)
            data = data.reshape(-1)
            collected_frames += frames_per_batch
            greedy_module.step(frames_per_batch)

            # if enable_mico:
            #     data = calculate_mico_distance(loss_module, data)

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

            # Warmup phase (due to the continue statement)
            # Additionally This help us to keep a track of the collected_frames
            # after the init_random_frames
            if collected_frames < init_random_frames:
                continue

            # optimization steps            
            sampled_tensordict = replay_buffer.sample(batch_size).to(device)

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
                if enable_mico:
                    priority = (1 - mico_priority_weight) * sampled_tensordict["td_error"]
                    priority += mico_priority_weight * sampled_tensordict["mico_distance"]
                else:
                    priority = sampled_tensordict["td_error"]

                if cfg.buffer.mico_priority.moving_average is not None:
                    # Update the moving average statistics
                    normalizer.update_statistics(priority)

                    # Normalize the tensor
                    priority = normalizer.normalize(priority)
                
                replay_buffer.update_priority(index=sampled_tensordict['index'], priority = priority)

            # NOTE: This is only one step (after n-updated steps defined before)
            # the target will update
            target_net_updater.step()

            # update weights of the inference policy
            # NOTE: Updates the policy weights if the policy of the data 
            # collector and the trained policy live on different devices.
            collector.update_policy_weights_()
        
            logger.log_per_step_info(data)



        # TODO: plot distributions of the mico distance in the replay buffer
        #       plot distributions of the td_error in the replay buffer
        # if cfg.logger.save_distributions:
        #     logger.log_info.update(
        #             {
        #                 "train/buffer_mico_distance_dist": wandb.Histogram(replay_buffer["mico_distance_metadata"].detach()),
        #             }
        #         )

        # TODO: After one iteration run evaluation if required
        # if logger.evaluation.enable:
        #     #

        training_time = time.time() - training_start
        average_steps_per_second =  logger.data["num_steps"] / training_time

        # After one iteration flush the information to wandb
        logger.log_info.update({"train/epsilon": greedy_module.eps,
                                "train/average_q_value": (data["action_value"] * data["action"]).detach().mean().item(),
                                "train/average_total_loss": loss["loss"].detach().mean().item(),
                                "train/average_td_loss": loss["td_loss"].detach().mean().item(),
                                "train/average_mico_loss": loss["mico_loss"].detach().mean().item() if enable_mico else 0,
                                "train/average_steps_per_second": average_steps_per_second,
                                })
        
        # this function will internally call log_per_iteration_info
        logger.flush_to_wandb(collected_frames)

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(execution_time)))
    print(f"Collected Frames: {collected_frames}, Total Episodes: {total_episodes}")
    print(f"Training took {formatted_time} (HH:MM:SS) to finish")
    print("Hyperparameters used:")
    print_hyperparameters(cfg)

    # TODO: Saved the model. Check how to save the model and load
    if cfg.logger.save_checkpoints:
        torch.save(model.state_dict(), f"outputs/models/{cfg.exp_name}_{cfg.env.env_name}_{date_str}.pt")


    wandb.finish()


if __name__ == "__main__":
    main()
