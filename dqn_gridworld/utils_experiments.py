import numpy as np
import torch
from tensordict import TensorDict
import utils_metric
from torchrl.envs.utils import step_mdp

# Create a matrix of all possible states taking into account that
# state is given by position of agent
# grid_size = 5

# visiting_count = torch.zeros((grid_size, grid_size))

# state_positions = data['observation']

# # Counting new visits
# visiting_count[state_positions[:, 0], state_positions[:, 1]] += 1

# # TODO: Save the visiting count at the end in a file
# # Save the visiting count
# torch.save(visiting_count, 'visiting_count.pt')

# Define a function to calculate the visiting frequency and get the entropy
def get_entropy(visiting_count):
    epsilon = 1e-6 # to avoid numerical issues
    visiting_frequency = visiting_count / visiting_count.sum()
    entropy = - (visiting_frequency * torch.log(visiting_frequency + epsilon)).sum()
    return entropy

# # Calculate the entropy
# entropy = get_entropy(visiting_count)

def get_distribution(data, bins = 64):
    hist = np.histogram(data, bins = bins)
    return hist, data.mean(), data.std()

# Calculate the distribution of bisimulation distance in the replay buffer
def bisimulation_distribution(experience_storage):
    choose_behavioral_distance = (experience_storage['behavioral_distance'] * experience_storage['action']).sum(axis=1)
    return get_distribution(choose_behavioral_distance)

def calculate_mico_distance(loss_module, tensordict):
    td_online_copy = tensordict.clone(False)
    with loss_module.value_network_params.to_module(loss_module.value_network):
        loss_module.value_network(td_online_copy)

    representations = td_online_copy['representation']

    # NOTE: In the code implementation, the author decided to compare the representations of 
    # the current states above vs all the representation of the current state but evaluated 
    # in the target_network.

    # Additionally, the next states passed throught the target network are needed for calculate
    # the target distance. Then, we are gonna pass the whole batch through the target network
    td_target_copy = tensordict.clone(False)
    with loss_module.target_value_network_params.to_module(loss_module.value_network):
        with torch.no_grad():
            loss_module.value_network(td_target_copy)
            loss_module.value_network(td_target_copy['next'])
            target_r = td_target_copy['representation'].detach()
            target_next_r = td_target_copy['next', 'representation'].detach()

    with torch.no_grad():

        # NOTE: IMPORTANT: Check if makes sense to compare online vs target, or only online
        # or only target
        if loss_module.priority_type == "current_vs_next":
            mico_distance = utils_metric.current_vs_next_mico_priorities(
                current_state_representations = representations, # online representation of current states
                next_state_representations = target_next_r, # target representation of next states
                mico_beta = loss_module.mico_beta)
        elif loss_module.priority_type == "all_vs_all":
            # It doesn't require new computations, only reshape and mean
            # Notice (as we are not collecting trajectories of two anymore)
            # We already calculate the distance, which is the online distance
            # NOTE: However there could be other variants as compare
            # the online vs online
            # or target vs target
            online_dist = utils_metric.representation_distances(
                        representations, target_r, loss_module.mico_beta)
                    
            mico_distance = utils_metric.all_vs_all_mico_priorities(
                        # first_batch_representation = representations,
                        # second_batch_representation = target_r,
                        batch_size = representations.shape[0],
                        mico_beta = loss_module.mico_beta,
                        distance_tensor=online_dist)
        else:
            raise ValueError("Invalid priority type")


    # TODO: I don't why an unsqueeze is needed
    mico_distance = mico_distance.unsqueeze(-1)

    if tensordict.device is not None:
        mico_distance = mico_distance.to(tensordict.device)

    tensordict.set(
        "mico_distance_metadata",
        mico_distance,
        inplace=True,
    )

    return tensordict

def calculate_td_error(loss_module, tensordict):
    td_copy = tensordict.clone(False)
    with loss_module.value_network_params.to_module(loss_module.value_network):
        loss_module.value_network(td_copy)

    action = tensordict.get(loss_module.tensor_keys.action)
    pred_val = td_copy.get(loss_module.tensor_keys.action_value)

    if loss_module.action_space == "categorical":
        if action.ndim != pred_val.ndim:
            # unsqueeze the action if it lacks on trailing singleton dim
            action = action.unsqueeze(-1)
        pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
    else:
        action = action.to(torch.float)
        pred_val_index = (pred_val * action).sum(-1)

    if loss_module.double_dqn:
        step_td = step_mdp(td_copy, keep_other=False)
        step_td_copy = step_td.clone(False)
        # Use online network to compute the action
        with loss_module.value_network_params.data.to_module(loss_module.value_network):
            loss_module.value_network(step_td)
            next_action = step_td.get(loss_module.tensor_keys.action)

        # Use target network to compute the values
        with loss_module.target_value_network_params.to_module(loss_module.value_network):
            loss_module.value_network(step_td_copy)
            next_pred_val = step_td_copy.get(loss_module.tensor_keys.action_value)

        if loss_module.action_space == "categorical":
            if next_action.ndim != next_pred_val.ndim:
                # unsqueeze the action if it lacks on trailing singleton dim
                next_action = next_action.unsqueeze(-1)
            next_value = torch.gather(next_pred_val, -1, index=next_action)
        else:
            next_value = (next_pred_val * next_action).sum(-1, keepdim=True)
    else:
        next_value = None
    target_value = loss_module.value_estimator.value_estimate(
        td_copy,
        target_params=loss_module.target_value_network_params,
        next_value=next_value,
    ).squeeze(-1)

    with torch.no_grad():
        priority_tensor = (pred_val_index - target_value).pow(2)
        priority_tensor = priority_tensor.unsqueeze(-1)
    if tensordict.device is not None:
        priority_tensor = priority_tensor.to(tensordict.device)

    tensordict.set(
        "td_error",
        priority_tensor,
        inplace=True,
    )

    return tensordict

def calculate_mico_distance_all_states(loss_module, tensordict):
    td_online_copy = tensordict.clone(False)
    with loss_module.value_network_params.to_module(loss_module.value_network):
        loss_module.value_network(td_online_copy)

    representations = td_online_copy['representation']

    # NOTE: In the code implementation, the author decided to compare the representations of 
    # the current states above vs all the representation of the current state but evaluated 
    # in the target_network.

    # Additionally, the next states passed throught the target network are needed for calculate
    # the target distance. Then, we are gonna pass the whole batch through the target network
    td_target_copy = tensordict.clone(False)
    with loss_module.target_value_network_params.to_module(loss_module.value_network):
        with torch.no_grad():
            loss_module.value_network(td_target_copy)
            target_r = td_target_copy['representation'].detach()

    with torch.no_grad():


        online_dist = utils_metric.representation_distances(
                        representations, target_r, loss_module.mico_beta)

    return online_dist

def get_all_states(test_env, policy=None):
    all_data_states = []

    for state in test_env.unwrapped._possible_states:
        data_state = test_env.reset(options = {"start_state": state})
        if policy:
            data_state = test_env.rollout(policy = policy, max_steps = 1)
        all_data_states.append(data_state)

    if policy is None:
        for state in test_env.unwrapped._targets_location:
            data_state = test_env.reset(options = {"start_state": state})
            all_data_states.append(data_state)

    keys2unsqueeze = [
        "behavioral_distance",
        "pixels",
        "observation",
        "episode_reward",
        "step_count",
        "terminated",
        "truncated"]

    tmp_dict = {}
    for key in all_data_states[0].keys():
        if key in keys2unsqueeze and policy is None:
            tmp_dict[key] = torch.cat([td[key].unsqueeze(0) for td in all_data_states], dim=0)
        else:
            tmp_dict[key] = torch.cat([td[key] for td in all_data_states], dim=0)

    concatenated_data = TensorDict(
        tmp_dict, 
        batch_size=torch.Size([len(all_data_states)]))
    
    return concatenated_data

def get_bisimulation_matrix(loss_module, test_env):

    all_states_tensordict = get_all_states(test_env)

    # Calculate the bisimulation distance of all vs all
    distances_all_vs_all = calculate_mico_distance_all_states(loss_module, all_states_tensordict)

    batch_size = len(all_states_tensordict)

    return distances_all_vs_all.reshape((batch_size,batch_size))

def get_priority_all_states(loss_module, test_env, policy, cfg_buffer, enable_mico):

    mico_priority_weight = cfg_buffer.mico_priority.priority_weight

    # Get all states
    all_states_tensordict = get_all_states(test_env, policy)
    
    # Calculate the td error of all states
    all_states_tensordict = calculate_td_error(loss_module, all_states_tensordict)

    # if enable_mico:
    #     # Calculate the mico distance of all states
    #     calculate_mico_distance(loss_module, all_states_tensordict)

    #     if cfg_buffer.mico_priority.normalize_priorities:
    #         norm_td_error = torch.log(all_states_tensordict["td_error"] + 1)
    #         norm_mico_distance = torch.log(all_states_tensordict["mico_distance_metadata"] + 1)
    #         priority = (1 - mico_priority_weight) * norm_td_error + mico_priority_weight * norm_mico_distance
    #     else:
    #         priority = (1 - mico_priority_weight) * all_states_tensordict["td_error"] + mico_priority_weight * all_states_tensordict["mico_distance_metadata"]
    # else:
    priority = all_states_tensordict["td_error"]

    # Normalize the td_error to get probabilities
    all_states_tensordict['priority'] = priority ** cfg_buffer.alpha
    all_states_tensordict['priority'] = priority / priority.max()

    return all_states_tensordict

def estimate_sampling_distribution(test_env, replay_buffer, device):

    # Get all possible states
    possible_states = np.array(list(test_env.unwrapped._possible_states))

    # Sample 3k of experiences
    # Concat the samples until reach 3k or larger
    sample = replay_buffer.sample()
    while len(sample) < 3000:
        tmp_sample = replay_buffer.sample()
        sample = torch.cat([sample, tmp_sample], dim=0)

    # Count the occurrences of each unique index pair
    unique_observations, counts = torch.unique(sample['observation'], return_counts=True, dim=0)
    unique_observations = unique_observations.cpu()
    counts = counts.to(device)

    # Get the counts of the 3000 samples
    visiting_count = torch.zeros((test_env.unwrapped.size, test_env.unwrapped.size), device=device)
    visiting_count[unique_observations[:, 0], unique_observations[:, 1]] += counts

    # Normalize the visiting count
    visiting_count = visiting_count / visiting_count.sum()

    # Access only to the states that are possible
    estimated_distribution = visiting_count[possible_states[:, 0], possible_states[:, 1]]

    return estimated_distribution

def estimate_on_policy_weight(test_env, replay_buffer, device):
    # Get all possible states
    possible_states = np.array(list(test_env.unwrapped._possible_states))

    # Get 3000 uniformely sampled indices from 0 to len(replay_buffer)
    indices = np.random.choice(len(replay_buffer), 3000)

    # Sample 3k of experiences
    sample = replay_buffer.storage[indices]

    # Count the occurrences of each unique index pair
    unique_observations, counts = torch.unique(sample['observation'], return_counts=True, dim=0)
    unique_observations = unique_observations.cpu()
    counts = counts.to(device)

    # Get the counts of the 3000 samples
    visiting_count = torch.zeros((test_env.unwrapped.size, test_env.unwrapped.size), device=device)
    visiting_count[unique_observations[:, 0], unique_observations[:, 1]] += counts

    # Normalize the visiting count
    visiting_count = visiting_count / visiting_count.sum()

    # Access only to the states that are possible
    estimated_distribution = visiting_count[possible_states[:, 0], possible_states[:, 1]]

    return estimated_distribution

def get_distances_distribution(loss_module,
                              test_env, 
                              policy, 
                              cfg_buffer, 
                              replay_buffer,
                              enable_mico,
                              device):

    priority_ideal_distribution = get_priority_all_states(loss_module, test_env, policy, cfg_buffer, enable_mico)

    estimated_distribution = estimate_sampling_distribution(test_env, replay_buffer, device)

    absolute_difference = torch.abs(priority_ideal_distribution['priority'] - estimated_distribution)
    distance_uniform = absolute_difference.mean()
    on_policy_weights = estimate_on_policy_weight(test_env, replay_buffer, device)
    distance_on_policy = (on_policy_weights * absolute_difference).mean()
    
    return distance_uniform, distance_on_policy
    
