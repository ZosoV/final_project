from torchrl.objectives import DQNLoss
from tensordict import TensorDict, TensorDictBase
import torch
from torch.nn.functional import huber_loss

import utils_metric
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
        td_loss = super().forward(tensordict)

        # Compute the MICODQN loss
        mico_loss = self.micodqn_loss(tensordict)

        total_loss = ((1. - self.mico_weight) * td_loss["loss"] + self.mico_weight * mico_loss)    

        td_out = TensorDict({"loss": total_loss, "td_loss": td_loss["loss"], "mico_loss": mico_loss}, [])

        return td_out

    def micodqn_loss(self, tensordict: TensorDictBase) -> TensorDict:
        # Compute the MICODQN loss

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
                # target_r = batch_target_representation[0::2]
                # target_next_r = batch_target_representation[1::2]

        # NOTE: the rewards are gotten from the next keys of the current states (even rows)
        rewards = td_online_copy['next','reward']

        online_dist = utils_metric.representation_distances(
        representations, target_r, self.mico_beta)

        # NOTE: Check the gradients requirement for the target distances OJO
        target_dist = utils_metric.target_distances(
            target_next_r, rewards, self.mico_gamma)
        
        # TODO: check if I need to use the vmap, if not use the other that
        # the library proposes
        # TODO: check what is hubber loss hahaha =D
        # mico_loss = torch.mean(torch.vmap(huber_loss)(online_dist,
        #                                                 target_dist))
        mico_loss = torch.mean(huber_loss(online_dist, target_dist))

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
        
        with torch.no_grad():

            # NOTE: IMPORTANT: Check if makes sense to compare online vs target, or only online
            # or only target
            if self.priority_type == "current_vs_next":
                mico_distance = utils_metric.current_vs_next_mico_priorities(
                    current_state_representations = representations, # online representation of current states
                    next_state_representations = target_next_r, # target representation of next states
                    mico_beta = self.mico_beta)
            elif self.priority_type == "all_vs_all":
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


        # TODO: I don't why an unsqueeze is needed
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
            if self.priority_type == "current_vs_next":
                mico_distance = utils_metric.current_vs_next_mico_priorities(
                    current_state_representations = representations, # online representation of current states
                    next_state_representations = target_next_r, # target representation of next states
                    mico_beta = self.mico_beta)
            elif self.priority_type == "all_vs_all":
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
                 enable_mico=True):
        super(MICODQNNetwork, self).__init__()

        self.activation_class = activation_class()
        self.use_batch_norm = use_batch_norm
        self.enable_mico = enable_mico
      
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