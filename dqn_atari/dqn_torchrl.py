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
import math

from tensordict.nn import TensorDictSequential
# from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
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
from utils_modules import MICODQNLoss, MovingAverageNormalization

import tempfile

from collections import deque

import numpy as np
np.float_ = np.float64

import gymnasium as gym
import ale_py
import os

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, default="config_torchrl")
# args = parser.parse_args()


@hydra.main(config_path=".", config_name="config_torchrl", version_base=None)
def main(cfg: "DictConfig"):

    # print("Using config file: ", args.config)

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
        torch.backends.cudnn.benchmark = True  # Speeds up convolution layers
        # torch.backends.cudnn.deterministic = False  # Allows non-deterministic but faster behavior

    # Set variables
    frames_per_batch = cfg.collector.frames_per_batch
    warmup_steps = cfg.collector.warmup_steps
    training_steps = cfg.collector.training_steps
    frame_stack = cfg.collector.frame_stack
    num_updates = cfg.loss.num_updates

    enable_grad_clipping = True if cfg.optim.max_grad_norm is not None else False
    max_grad = cfg.optim.max_grad_norm
    batch_size = cfg.buffer.batch_size
    enable_prioritized_replay = cfg.buffer.prioritized_replay.enable

    mico_priority_weight = cfg.buffer.prioritized_replay.priority_weight
    enable_mico = cfg.loss.mico_loss.enable
    priority_type = cfg.buffer.prioritized_replay.priority_type

    # Evaualation
    enable_evaluation = cfg.logger.evaluation.enable
    eval_freq = cfg.logger.evaluation.eval_freq

    # Set the logger
    # Set the name with the data of the experiment
    current_date = datetime.datetime.now()
    date_str = current_date.strftime("%Y_%m_%d-%H_%M_%S")  # Includes date and time
    run_name = f"{cfg.run_name}_{date_str}"
    summary_writing_frequency = cfg.logger.summary_writing_frequency

    # Initialize W&B run with config
    hyperparameters = {
        "game" : cfg.env.env_name,
        "priority_type" : cfg.buffer.prioritized_replay.priority_type,
        "priority_weight" : cfg.buffer.prioritized_replay.priority_weight,
        "moving_average" : cfg.buffer.prioritized_replay.moving_average,
        "alpha" : cfg.buffer.prioritized_replay.alpha,
        "beta" : cfg.buffer.prioritized_replay.beta,
        "epsilon_decay_period" : cfg.collector.epsilon_decay_period,
        "epsilon_end" : cfg.collector.epsilon_end,
        "frames_per_batch" : cfg.collector.frames_per_batch,
        "warmup_steps" : cfg.collector.warmup_steps,
        "buffer_size" : cfg.buffer.buffer_size,
        "batch_size" : cfg.buffer.batch_size,
        "target_update_period" : cfg.loss.target_update_period,
        "gamma" : cfg.loss.gamma,
        "mico_beta" : cfg.loss.mico_loss.mico_beta,
        "mico_gamma" : cfg.loss.mico_loss.mico_gamma,
        "hydra_cfg" : dict(cfg)
    }

    # Create path to save logs
    logs_path = f"outputs/{cfg.run_name}"
    os.makedirs(logs_path, exist_ok=True)

    wandb.init(
        name=run_name,
        project=cfg.project_name, 
        group=cfg.group_name, 
        mode=cfg.logger.mode, 
        config=hyperparameters,  # Pass the entire config for hyperparameter tracking
        dir=logs_path
    )

    device = cfg.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    device_steps = cfg.running_setup.device_steps

    # Print the current seed and group
    print(f"Running with Seed: {seed} on Device: {device}")

    # Make the components
    # Policy
    model = make_dqn_model(cfg.env.env_name, 
                           cfg.policy, 
                           frame_stack,
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


    # Flushing the first epsilon value
    wandb.log({"train/epsilon": greedy_module.eps.item()}, step=0)

    env_maker = lambda: make_env(cfg.env.env_name,
                                frame_stack = frame_stack,
                                device = device_steps, 
                                seed = cfg.env.seed,
                                max_steps_per_episode = cfg.collector.max_steps_per_episode)
    # Create the collector
    if cfg.running_setup.num_envs == 1:
    # NOTE: warmup_steps: Number of frames 
    # for which the policy is ignored before it is called.
        print("Using Single SyncDataCollector")
        collector = SyncDataCollector(
            create_env_fn=env_maker,
            policy=model_explore,
            frames_per_batch=frames_per_batch,
            exploration_type=ExplorationType.RANDOM,
            env_device=device_steps,
            storing_device=device_steps,
            policy_device=device_steps,
            split_trajs=False,
            init_random_frames=warmup_steps,
        )
    else:
        print("Using MultiSyncDataCollector")
        collector = MultiSyncDataCollector(
            create_env_fn=[env_maker] * cfg.running_setup.num_envs,
            policy=model_explore,
            frames_per_batch=frames_per_batch,
            exploration_type=ExplorationType.RANDOM,
            env_device=device_steps,
            policy_device=device_steps,
            storing_device=device_steps,
            split_trajs=False,
            init_random_frames=warmup_steps,
            cat_results="stack",
            num_threads = cfg.running_setup.num_threads     
        )

    # Create the replay buffer
    if enable_prioritized_replay:
        print("Using Prioritized Replay Buffer")
        sampler = PrioritizedSampler(
            max_capacity=cfg.buffer.buffer_size, 
            alpha=cfg.buffer.prioritized_replay.alpha, 
            beta=cfg.buffer.prioritized_replay.beta)
    else:
        sampler = RandomSampler()

    if cfg.buffer.scratch_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        scratch_dir = tempdir.name
    else:
        scratch_dir = cfg.buffer.scratch_dir

    # Create a dir in TMPDIR/<run_name>/ to store the replay buffer
    # scratch_dir = os.path.join(os.environ["EXP_BUFF"], f"rb_{run_name}")
    # os.makedirs(scratch_dir, exist_ok=True)

    print(f"Using scratch_dir: {scratch_dir}")

    if cfg.running_setup.enable_lazy_tensor_buffer:
        storage = LazyTensorStorage(
            max_size=cfg.buffer.buffer_size,
            # device=device  # Important: Ensures data is stored directly in RAM
        )
    else:
        storage = LazyMemmapStorage( # NOTE: additional line
                max_size=cfg.buffer.buffer_size,
                scratch_dir=scratch_dir,
                # device = device_steps
            )

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=cfg.running_setup.pin_memory,
        prefetch=cfg.running_setup.prefetch if cfg.running_setup.prefetch is not None else None,
        storage=storage,
        batch_size=cfg.buffer.batch_size,
        sampler = sampler
    )

    # Create a moving average for the priorities
    if cfg.buffer.prioritized_replay.moving_average is not None:
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
            priority_type=priority_type,
        )
    else:
        loss_module = DQNLoss(
            value_network=model,
            loss_function="l2", 
            delay_value=True, # delay_value=True means we will use a target network
        )
    
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
    if enable_evaluation:
        test_env = make_env(cfg.env.env_name,
                            frame_stack,
                            device,
                            seed=cfg.env.seed,
                            max_steps_per_episode = cfg.collector.max_steps_per_episode, 
                            is_test=True)
        test_env.eval()

    # Main loop
    steps_so_far = 0 # Also corresponds to the current step
    total_episodes = 0
    start_iteration = 0

    if cfg.logger.load_checkpoint:
        # Load checkpoint
        checkpoint = torch.load(cfg.logger.load_checkpoint_path)

        # Restore model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore other states
        start_iteration = checkpoint['iteration'] + 1
        steps_so_far = checkpoint['steps_so_far']
        total_episodes = checkpoint['total_episodes']

        # Restore replay buffer if applicable
        # if 'replay_buffer' in checkpoint:
        #     replay_buffer.load_state_dict(checkpoint['replay_buffer'])

        # # # Restore exploration module
        # if 'greedy_module_state_dict' in checkpoint:
        #     greedy_module.load_state_dict(checkpoint['greedy_module_state_dict'])


    start_time = time.time()
    c_iter = iter(collector)

    # Set thread before the loop start:
    if cfg.running_setup.num_threads is not None:
        print(f"Threads before setting manually: {torch.get_num_threads()}")
        torch.set_num_threads(cfg.running_setup.num_threads)  
        print(f"Threads after setting manually: {torch.get_num_threads()}")

    # Running the warmup phase to fill the replay buffer
    print("Warmup phase ...")
    curr_warmup_steps = 0
    for i in range(warmup_steps // frames_per_batch):
        data = next(c_iter)
        data = data.reshape(-1)
        curr_warmup_steps += data.numel()
        replay_buffer.extend(data)
    print("Warmup phase finished with steps: ", curr_warmup_steps)

    print("Initial PyTorch Threads: ", torch.get_num_threads())
    print("Init main loop ...")

    # avg_iter_time = 0

    for iteration in range(start_iteration, cfg.collector.num_iterations + start_iteration):

        sum_return = 0.0
        number_of_episodes = 0
        num_steps = 0

        sum_score = 0

        # steps_in_batch = math.ceil(training_steps / frames_per_batch)
        steps_in_batch = training_steps // frames_per_batch
        data_iter = tqdm.tqdm(
                desc="IT_%s:%d" % ("train", iteration),
                total= steps_in_batch * frames_per_batch,
                bar_format="{l_bar}{r_bar}"
            )
        
        # training_start = time.time()
        
        iter_steps = 0
        for i in range(steps_in_batch):
            data = next(c_iter)

            current_env_step = data.numel()
            data_iter.update(current_env_step)

            # NOTE: This reshape must be for frame data (maybe)
            data = data.reshape(-1)
            steps_so_far += current_env_step
            iter_steps += current_env_step
            
            greedy_module.step(current_env_step)  # 4 is the skip frame

            # Flusing the epsilon for checking
            # wandb.log({"train/epsilon": greedy_module.eps.item()}, step=steps_so_far * 4)

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

            # Update the statistics
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            # When there are at least one done trajectory in the data batch
            if len(episode_rewards) > 0:
                # Check that I'm only using one episode
                assert len(episode_rewards) == 1
                sum_return += episode_rewards.sum().item()
                number_of_episodes += len(episode_rewards)

                sum_score += data["next", "episode_score"][data["next", "done"]].sum().item()
                
                # Get episode num_steps
                episode_num_steps = data["next", "step_count"][data["next", "done"]]
                num_steps += episode_num_steps.sum().item()

            # Warmup phase (due to the continue statement)
            # Additionally This help us to keep a track of the steps_so_far
            # after the warmup_steps
            # if steps_so_far < warmup_steps:
            #     continue

            # optimization steps
            for j in range(num_updates):            
                sampled_tensordict = replay_buffer.sample(batch_size).to(device)#, non_blocking=True)

                # Also the loss module will use the current and target model to get the q-values
                loss = loss_module(sampled_tensordict)
                q_loss = loss["loss"]
                optimizer.zero_grad()
                q_loss.backward()
                if enable_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        list(loss_module.parameters()), max_norm=max_grad
                    )
                optimizer.step()

                # Update the priorities
                if enable_prioritized_replay:
                    if enable_mico and priority_type != "PER":
                        priority = (1 - mico_priority_weight) * sampled_tensordict["td_error"]
                        priority += mico_priority_weight * sampled_tensordict["mico_distance"]
                    else:
                        priority = sampled_tensordict["td_error"]

                    if cfg.buffer.prioritized_replay.moving_average is not None:
                        # Update the moving average statistics
                        normalizer.update_statistics(priority)

                        # Normalize the tensor
                        priority = normalizer.normalize(priority)
                    
                    replay_buffer.update_priority(index=sampled_tensordict['index'], priority = priority)

                target_net_updater.step()

            # update weights of the inference policy
            # NOTE: Updates the policy weights if the policy of the data 
            # collector and the trained policy live on different devices.
            collector.update_policy_weights_()

            if iter_steps > 0 and iter_steps % summary_writing_frequency == 0:

                # training_time = time.time() - training_start
                # average_steps_per_second =  num_steps / training_time
                # avg_iter_time += training_time


                # if steps_so_far >= warmup_steps:
                info2flush = {
                    "train/epsilon": greedy_module.eps.item(),
                    "train/average_q_value": torch.gather(data["action_value"], 1, data["action"].unsqueeze(1)).mean().item(),
                    # "train/average_steps_per_second": average_steps_per_second,
                    # "train/iter_time" : training_time,
                    # "train/avg_iter_time" : avg_iter_time / (iteration + 1),
                    "train/average_total_loss": loss["loss"].mean().item(),
                    "train/average_td_loss": loss["loss"].mean().item(),
                }
                
                if enable_mico:
                    info2flush.update({
                        "train/average_mico_loss": loss["mico_loss"].mean().item(),
                        "train/average_td_loss": loss["td_loss"].mean().item(),
                    })
            
                if number_of_episodes > 0:
                    total_episodes += number_of_episodes
                    info2flush["train/average_return"] = sum_return / number_of_episodes
                    info2flush["train/average_score"] = sum_score / number_of_episodes
                    info2flush["train/average_episode_length"] = num_steps / number_of_episodes

                # Flush the information to wandb
                # NOTE: We must flush the information by multiplying the step by 4 because
                # the skip frame is 4 in the environment. Then the collected frames are 4 times
                # print("Average Score: ", sum_score / number_of_episodes)
                wandb.log(info2flush, step=steps_so_far * 4)

                # Set values to default values
                sum_return = 0
                number_of_episodes = 0
                num_steps = 0
                sum_score = 0
            

        # Evaluation
        # if enable_evaluation:
        #     if (iteration + 1) % eval_freq == 0:
        #         test_reward, accuracy = eval_model(model, test_env, iteration)
        #         info2flush["eval/average_return"] = test_reward
        #         info2flush["eval/accuracy"] = accuracy


        if cfg.logger.save_checkpoint:
            if (iteration + 1) % cfg.logger.save_checkpoint_freq == 0:
                print(f"Saving checkpoint at iteration {iteration}")
                
                # Create the directory
                path = f"models/dqn/{date_str}"
                os.makedirs(path, exist_ok=True)
                
                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration,
                    'steps_so_far': steps_so_far,
                    'total_episodes': total_episodes,
                    # 'replay_buffer': replay_buffer.state_dict(),  # Save replay buffer state if supported
                    # 'greedy_module_state_dict': greedy_module.state_dict(),
                }
                torch.save(checkpoint, f"{path}/{cfg.run_name}_checkpoint_{iteration}.pth")



    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(execution_time)))
    print(f"Collected Frames: {steps_so_far * 4}, Total Episodes: {total_episodes}")
    print(f"Training took {formatted_time} (HH:MM:SS) to finish")
    print("Hyperparameters used:")
    print_hyperparameters(cfg)

    # TODO: Saved the model. Check how to save the model and load
    if cfg.logger.save_checkpoint:
        print("Saving the final model")
        path = f"models/dqn/{date_str}"
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), f"{path}/{cfg.run_name}_model.pth")

        

    wandb.finish()


if __name__ == "__main__":
    main()