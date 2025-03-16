from torchrl.objectives import DQNLoss
from tensordict import TensorDict, TensorDictBase
import torch
from torch.nn.functional import huber_loss
from collections import deque

import utils_metric
import wandb
import numpy as np
from torchrl.envs.utils import step_mdp
from torchrl.objectives.utils import (
    _reduce,
    distance_loss,
)
# from . import utils_metric

class MICODQNLoss(DQNLoss):

    def __init__(self, *args, 
                 mico_weight=0.5, 
                 mico_gamma=0.99, 
                 mico_beta=0.1, 
                 priority_type="all_vs_all", 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mico_weight = mico_weight
        self.mico_gamma = mico_gamma
        self.mico_beta = mico_beta
        self.priority_type = priority_type


    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        # Compute the loss
        # td_loss = super().forward(tensordict)
        td_loss, td_copy = self.dqn_forward(tensordict)

        # Compute the MICODQN loss
        mico_loss = self.mico_forward(tensordict, td_copy)

        total_loss = ((1. - self.mico_weight) * td_loss["loss"] + self.mico_weight * mico_loss)    

        return TensorDict({"loss": total_loss, "td_loss": td_loss["loss"], "mico_loss": mico_loss}, [])
    
    def dqn_forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the DQN loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the DQN loss.

        """
        td_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            if action.ndim != pred_val.ndim:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1)

        if self.double_dqn:
            step_td = step_mdp(td_copy, keep_other=False)
            step_td_copy = step_td.clone(False)
            # Use online network to compute the action
            with self.value_network_params.data.to_module(self.value_network):
                self.value_network(step_td)
                next_action = step_td.get(self.tensor_keys.action)

            # Use target network to compute the values
            with self.target_value_network_params.to_module(self.value_network):
                self.value_network(step_td_copy)
                next_pred_val = step_td_copy.get(self.tensor_keys.action_value)

            if self.action_space == "categorical":
                if next_action.ndim != next_pred_val.ndim:
                    # unsqueeze the action if it lacks on trailing singleton dim
                    next_action = next_action.unsqueeze(-1)
                next_value = torch.gather(next_pred_val, -1, index=next_action)
            else:
                next_value = (next_pred_val * next_action).sum(-1, keepdim=True)
        else:
            next_value = None
        target_value = self.value_estimator.value_estimate(
            td_copy,
            target_params=self.target_value_network_params,
            next_value=next_value,
        ).squeeze(-1)

        with torch.no_grad():
            priority_tensor = (pred_val_index - target_value).pow(2)
            priority_tensor = priority_tensor.unsqueeze(-1)
        if tensordict.device is not None:
            priority_tensor = priority_tensor.to(tensordict.device)

        tensordict.set(
            self.tensor_keys.priority,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        loss = _reduce(loss, reduction=self.reduction)
        td_out = TensorDict({"loss": loss}, [])
    
        return td_out, td_copy

    def mico_forward(self, tensordict: TensorDictBase, tensordict_copy: TensorDictBase) -> TensorDict:
        # Compute the MICODQN loss

        td_online_copy = tensordict_copy
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_online_copy)

        representations = td_online_copy.get('representation').clone()

        # NOTE: In the code implementation, the author decided to compare the representations of 
        # the current states above vs all the representation of the current state but evaluated 
        # in the target_network.

        # Additionally, the next states passed throught the target network are needed for calculate
        # the target distance. Then, we are gonna pass the whole batch through the target network
        td_target_copy = tensordict_copy
        with self.target_value_network_params.to_module(self.value_network):
            with torch.no_grad():
                self.value_network(td_target_copy)
                self.value_network(td_target_copy['next'])
                target_r = td_target_copy.get('representation') #.detach()
                target_next_r = td_target_copy.get(('next', 'representation')) #.detach()
                # target_r = batch_target_representation[0::2]
                # target_next_r = batch_target_representation[1::2]

        # NOTE: the rewards are gotten from the next keys of the current states (even rows)
        rewards = td_online_copy.get(('next','reward'))

        online_dist = utils_metric.representation_distances(
        representations, target_r, self.mico_beta)

        # NOTE: Check the gradients requirement for the target distances OJO
        target_dist = utils_metric.target_distances(
            target_next_r, rewards, self.mico_gamma)
        
        # TODO: check if I need to use the vmap, if not use the other that
        # the library proposes
        # TODO: check what is hubber loss hahaha =D
        mico_loss = torch.mean(torch.vmap(huber_loss)(online_dist,
                                                        target_dist))
        # mico_loss = torch.mean(huber_loss(online_dist, target_dist))

        # MICO Priority Calculation

        # NOTE: online distances calculates the distances all vs all on the current batch 
        # by taking the online_representation and target_representation. This distance metric 
        # is our best approximation to behavioral similarity, which can be used as a surrogate
        # for the priority.

        # However for example, if I have a batch of 32, the online distance will return a tensor
        # of 32x32=1024 distances, and we want to assign a priority only to the initial 32 states
        # so we are gonna take the average on a window of 32
        # Additioanlly, notice that we are not taking into acount the next states, we have to do the
        # same with the next states. So that, it's better to get the target of the whole batch
        # Remember that the even rows are the current states and the odd rows are the next states
        # But here we are gonna use all the batch.

        # NOTE: Important: It seems that makes more sense to use the mico metric taking into account
        # the rewards. Check if it improves or not the performance
        # I hypothesis that it's not gonna a have major impact because the rewards are not gonna be
        # very different between the current and next states, but it's worth to check

        # I even could use the mico error as the priority, but I think it's better to use the distance
        
        if self.priority_type != "PER":
            with torch.no_grad():

                # NOTE: IMPORTANT: Check if makes sense to compare online vs target, or only online
                # or only target
                if self.priority_type == "BPERcn":
                    mico_distance = utils_metric.current_vs_next_mico_priorities(
                        current_state_representations = representations, # online representation of current states
                        next_state_representations = target_next_r, # target representation of next states
                        mico_beta = self.mico_beta)
                elif self.priority_type == "BPERaa":
                    # It doesn't require new computations, only reshape and mean
                    # Notice (as we are not collecting trajectories of two anymore)
                    # We already calculate the distance, which is the online distance
                    # NOTE: However there could be other variants as compare
                    # the online vs online
                    # or target vs target
                    mico_distance = utils_metric.all_vs_all_mico_priorities(
                                # first_batch_representation = representations,
                                # second_batch_representation = target_r,
                                batch_size = representations.shape[0],
                                mico_beta = self.mico_beta,
                                distance_tensor=online_dist)
                else:
                    raise ValueError("Invalid priority type")


            # TODO: I don't know why I need to an unsqueeze is needed
            mico_distance = mico_distance.unsqueeze(-1)

            if tensordict.device is not None:
                mico_distance = mico_distance.to(tensordict.device)

            tensordict.set(
                "mico_distance",
                mico_distance,
                inplace=True,
            )

        return mico_loss
    
    def calculate_mico_distance(self, tensordict: TensorDictBase) -> TensorDict:
        td_online_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_online_copy)

        representations = td_online_copy['representation']

        # NOTE: In the code implementation, the author decided to compare the representations of 
        # the current states above vs all the representation of the current state but evaluated 
        # in the target_network.

        # Additionally, the next states passed throught the target network are needed for calculate
        # the target distance. Then, we are gonna pass the whole batch through the target network
        td_target_copy = tensordict.clone(False)
        with self.target_value_network_params.to_module(self.value_network):
            with torch.no_grad():
                self.value_network(td_target_copy)
                self.value_network(td_target_copy['next'])
                target_r = td_target_copy['representation'].detach()
                target_next_r = td_target_copy['next', 'representation'].detach()

        with torch.no_grad():

            # NOTE: IMPORTANT: Check if makes sense to compare online vs target, or only online
            # or only target
            if self.priority_type == "BPERcn":
                mico_distance = utils_metric.current_vs_next_mico_priorities(
                    current_state_representations = representations, # online representation of current states
                    next_state_representations = target_next_r, # target representation of next states
                    mico_beta = self.mico_beta)
            elif self.priority_type == "BPERaa":
                # It doesn't require new computations, only reshape and mean
                # Notice (as we are not collecting trajectories of two anymore)
                # We already calculate the distance, which is the online distance
                # NOTE: However there could be other variants as compare
                # the online vs online
                # or target vs target
                online_dist = utils_metric.representation_distances(
                            representations, target_r, self.mico_beta)
                        
                mico_distance = utils_metric.all_vs_all_mico_priorities(
                            # first_batch_representation = representations,
                            # second_batch_representation = target_r,
                            batch_size = representations.shape[0],
                            mico_beta = self.mico_beta,
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
    
class MICODQNNetwork(torch.nn.Module):
    """The convolutional network used to compute the agent's Q-values."""
    def __init__(self, 
                 input_shape,
                 num_outputs,
                 num_cells_cnn, 
                 kernel_sizes, 
                 strides, 
                 num_cells_mlp,
                 activation_class,
                 use_batch_norm=False,
                 enable_mico=False):
        super(MICODQNNetwork, self).__init__()

        self.enable_mico = enable_mico

        self.activation_class = activation_class()
        self.use_batch_norm = use_batch_norm
      
        # Input shape example: (4, 84, 84)
        channels, width, height = input_shape
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

    def forward(self, input):

        x = input
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)

            # NOTE: The collector uses a tensor for checking something
            # but this tensor is not in batch format, so we need to
            # check if the tensor is in batch format to apply batch norm
            if self.use_batch_norm and len(input.shape) == 4:
                x = self.batch_norm_layers[i](x)
            x = self.activation_class(x)

        if len(input.shape) == 4:
            x = x.view(x.size(0), -1)  # Flatten the tensor
        else:
            x = x.view(-1)

        # NOTE: I don't know if take it before or after 
        # the activation
        representation = x

        if self.fc_layers is not None:
            for fc_layer in self.fc_layers:
                x = self.activation_class(fc_layer(x))
        q_values = self.output_layer(x)

        if self.enable_mico:
            return q_values, representation
        else:
            return q_values
    
class MovingAverageNormalization:
    def __init__(self, momentum=0.01, epsilon=1e-4):
        """
        Initialize the MovingAverageNormalization.
        
        Args:
        - momentum: The momentum for moving average update. (0 < momentum < 1)
        - epsilon: A small value to prevent division by zero.
        """
        self.momentum = momentum
        self.epsilon = epsilon
        self.mean = None
        self.std = None
    
    def update_statistics(self, tensor):
        """
        Update the moving mean and standard deviation using the input tensor.
        
        Args:
        - tensor: Input tensor of shape [batch_size, 1] or similar.
        """
        current_mean = tensor.mean(dim=0, keepdim=True)
        current_std = tensor.std(dim=0, keepdim=True)

        if self.mean is None or self.std is None:
            # Initialize mean and std with the first batch's statistics
            self.mean = current_mean
            self.std = current_std
        else:
            # Update moving average mean and std
            self.mean = self.momentum * current_mean + (1 - self.momentum) * self.mean
            self.std = self.momentum * current_std + (1 - self.momentum) * self.std

    def normalize(self, tensor):
        """
        Normalize the tensor using the moving average statistics.
        
        Args:
        - tensor: Input tensor of shape [batch_size, 1] or similar.
        
        Returns:
        - Normalized tensor.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Statistics must be initialized by calling `update_statistics` first.")
        
        return (tensor - self.mean) / (self.std + self.epsilon)
    