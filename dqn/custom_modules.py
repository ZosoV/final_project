from torchrl.objectives import DQNLoss
from tensordict import TensorDict, TensorDictBase
import torch
from torch.nn.functional import huber_loss

import metric_utils
# from . import metric_utils

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

        total_loss = ((1. - self.mico_weight) * td_loss["td_loss"] + self.mico_weight * mico_loss)    

        td_out = TensorDict({"loss": total_loss, "td_loss": td_loss["td_loss"], "mico_loss": mico_loss}, [])

        return td_out

    def micodqn_loss(self, tensordict: TensorDictBase) -> TensorDict:
        # Compute the MICODQN loss

        td_online_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_online_copy)

        # NOTE: Take only the even rows to only consider the current state not the next state
        representations = td_online_copy['representation'][0::2]

        # NOTE: In the code implementation, the author decided to compare the representations of 
        # the current states above vs all the representation of the current state but evaluated 
        # in the target_network.

        # Additionally, the next states are also evaluated in the target network. Then, we are gonna
        # pass the observations but now using the target network to get these representations

        td_target_copy = tensordict.clone(False)
        with self.target_value_network_params.to_module(self.value_network):
            with torch.no_grad():
                self.value_network(td_target_copy)
                batch_target_representation = td_target_copy['representation'].detach()
                target_r = batch_target_representation[0::2]
                target_next_r = batch_target_representation[1::2]

        # NOTE: the rewards are gotten from the next keys of the current states (even rows)
        rewards = td_online_copy['next','reward'][0::2]

        online_dist = metric_utils.representation_distances(
        representations, target_r, self.mico_beta)

        # NOTE: Check the gradients requirement for the target distances OJO
        target_dist = metric_utils.target_distances(
            target_next_r, rewards, self.mico_gamma)
        
        # TODO: check if I need to use the vmap, if not use the other that
        # the library proposes
        # TODO: check what is hubber loss hahaha =D
        # mico_loss = torch.mean(torch.vmap(huber_loss)(online_dist,
        #                                                 target_dist))
        mico_loss = torch.mean(huber_loss(online_dist, target_dist))

        # TODO: Calculate the mico priority

        # NOTE: online distance calculates the distances all vs all on the current batch 
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
        
        with torch.no_grad():

            # NOTE: IMPORTANT: Check if makes sense to compare online vs target, or only online
            # or only target
            if self.priority_type == "current_vs_next":
                mico_distance = metric_utils.current_vs_next_mico_priorities(
                    current_state_representations = representations, # online representation of current states
                    next_state_representations = target_next_r, # target representation of next states
                    mico_beta = self.mico_beta)
            elif self.priority_type == "all_vs_all":
                # It doesn't require new computations, only reshape and mean
                mico_distance = metric_utils.all_vs_all_mico_priorities(
                            first_batch_representation = td_online_copy['representation'],
                            second_batch_representation = batch_target_representation,
                            mico_beta = self.mico_beta)
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