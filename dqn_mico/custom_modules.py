import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey
from torch import nn
from torchrl.data.tensor_specs import TensorSpec

from torchrl.data.utils import _find_action_space

from torchrl.envs.utils import step_mdp
from torchrl.modules.tensordict_module.actors import (
    DistributionalQValueActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torch.nn.functional import huber_loss
from torchrl.objectives.value import TDLambdaEstimator
from torchrl.objectives.value.advantages import TD0Estimator, TD1Estimator

import metric_utils
# from . import metric_utils


class MICODQNLoss(LossModule):
    """The DQN Loss class.

    Args:
        value_network (QValueActor or nn.Module): a Q value operator.

    Keyword Args:
        loss_function (str, optional): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
            Defaults to "l2".
        delay_value (bool, optional): whether to duplicate the value network
            into a new target value network to
            create a DQN with a target network. Default is ``False``.
        double_dqn (bool, optional): whether to use Double DQN, as described in
            https://arxiv.org/abs/1509.06461. Defaults to ``False``.
        action_space (str or TensorSpec, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult_one_hot"``, ``"binary"`` or ``"categorical"``,
            or an instance of the corresponding specs (:class:`torchrl.data.OneHotDiscreteTensorSpec`,
            :class:`torchrl.data.MultiOneHotDiscreteTensorSpec`,
            :class:`torchrl.data.BinaryDiscreteTensorSpec` or :class:`torchrl.data.DiscreteTensorSpec`).
            If not provided, an attempt to retrieve it from the value network
            will be made.
        priority_key (NestedKey, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            The key at which priority is assumed to be stored within TensorDicts added
            to this ReplayBuffer.  This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.  Defaults to ``"td_error"``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.

    Examples:
        >>> from torchrl.modules import MLP
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> n_obs, n_act = 4, 3
        >>> value_net = MLP(in_features=n_obs, out_features=n_act)
        >>> spec = OneHotDiscreteTensorSpec(n_act)
        >>> actor = QValueActor(value_net, in_keys=["observation"], action_space=spec)
        >>> loss = MICODQNLoss(actor, action_space=spec)
        >>> batch = [10,]
        >>> data = TensorDict({
        ...     "observation": torch.randn(*batch, n_obs),
        ...     "action": spec.rand(batch),
        ...     ("next", "observation"): torch.randn(*batch, n_obs),
        ...     ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...     ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...     ("next", "reward"): torch.randn(*batch, 1)
        ... }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                loss: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["observation", "next_observation", "action", "next_reward", "next_done", "next_terminated"]``,
    and a single loss value is returned.

    Examples:
        >>> from torchrl.objectives import MICODQNLoss
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torch import nn
        >>> import torch
        >>> n_obs = 3
        >>> n_action = 4
        >>> action_spec = OneHotDiscreteTensorSpec(n_action)
        >>> value_network = nn.Linear(n_obs, n_action) # a simple value model
        >>> dqn_loss = MICODQNLoss(value_network, action_space=action_spec)
        >>> # define data
        >>> observation = torch.randn(n_obs)
        >>> next_observation = torch.randn(n_obs)
        >>> action = action_spec.rand()
        >>> next_reward = torch.randn(1)
        >>> next_done = torch.zeros(1, dtype=torch.bool)
        >>> next_terminated = torch.zeros(1, dtype=torch.bool)
        >>> loss_val = dqn_loss(
        ...     observation=observation,
        ...     next_observation=next_observation,
        ...     next_reward=next_reward,
        ...     next_done=next_done,
        ...     next_terminated=next_terminated,
        ...     action=action)

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the chosen action value is expected.
                Will be used for the underlying value estimator. Defaults to ``"chosen_action_value"``.
            action_value (NestedKey): The input tensordict key where the action value is expected.
                Defaults to ``"action_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "chosen_action_value"
        action_value: NestedKey = "action_value"
        action: NestedKey = "action"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    out_keys = ["loss"]

    def __init__(
        self,
        value_network: Union[QValueActor, nn.Module],
        *,
        loss_function: Optional[str] = "l2",
        delay_value: bool = True,
        double_dqn: bool = False,
        gamma: float = None,
        action_space: Union[str, TensorSpec] = None,
        priority_key: str = None,
        reduction: str = None,
        mico_weight: float = 0.5,
        mico_gamma: float = 0.99,
        mico_beta: float = 0.1,
        priority_type: str = "all_vs_all",
        priority_alpha: float = 0.6,
    ) -> None:
        if reduction is None:
            reduction = "mean"
        super().__init__()

        self._in_keys = None
        if double_dqn and not delay_value:
            raise ValueError("double_dqn=True requires delay_value=True.")
        self.double_dqn = double_dqn
        self._set_deprecated_ctor_keys(priority=priority_key)
        self.delay_value = delay_value
        value_network = ensure_tensordict_compatible(
            module=value_network,
            wrapper_type=QValueActor,
            action_space=action_space,
        )

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        self.value_network_in_keys = value_network.in_keys

        self.loss_function = loss_function
        if action_space is None:
            # infer from value net
            try:
                action_space = value_network.spec
            except AttributeError:
                # let's try with action_space then
                try:
                    action_space = value_network.action_space
                except AttributeError:
                    raise ValueError(
                        "The action space could not be retrieved from the value_network. "
                        "Make sure it is available to the DQN loss module."
                    )
        if action_space is None:
            warnings.warn(
                "action_space was not specified. MICODQNLoss will default to 'one-hot'."
                "This behaviour will be deprecated soon and a space will have to be passed."
                "Check the MICODQNLoss documentation to see how to pass the action space. "
            )
            action_space = "one-hot"
        self.action_space = _find_action_space(action_space)
        self.reduction = reduction

        self.mico_weight = mico_weight
        self.mico_gamma = mico_gamma
        self.mico_beta = mico_beta
        self.priority_type = priority_type
        
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        


    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.value_network.in_keys,
            *[("next", key) for key in self.value_network.in_keys],
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(**hp, value_network=self.value_network)
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(**hp, value_network=self.value_network)
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp, value_network=self.value_network
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value_target": self.tensor_keys.value_target,
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)


    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the DQN loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the DQN loss.

        """
        # TODO: This will help us to debug the code, but it is better to
        # inherit from QLoss and only update this funciton accordingly, instead of
        # writing the whole class again.
        td_online_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_online_copy)

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_online_copy.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            if action.ndim != pred_val.ndim:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1)

        if self.double_dqn:
            step_td = step_mdp(td_online_copy, keep_other=False)
            step_td_online_copy = step_td.clone(False)
            # Use online network to compute the action
            with self.value_network_params.data.to_module(self.value_network):
                self.value_network(step_td)
                next_action = step_td.get(self.tensor_keys.action)

            # Use target network to compute the values
            with self.target_value_network_params.to_module(self.value_network):
                self.value_network(step_td_online_copy)
                next_pred_val = step_td_online_copy.get(self.tensor_keys.action_value)

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
            td_online_copy,
            target_params=self.target_value_network_params,
            next_value=next_value,
        ).squeeze(-1)

        ## Metric calculation

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
                    mico_beta = self.mico_beta).unsqueeze(-1)
            elif self.priority_type == "all_vs_all":
                # It doesn't require new computations, only reshape and mean
                mico_distance = metric_utils.all_vs_all_mico_priorities(
                            first_batch_representation = td_online_copy['representation'],
                            second_batch_representation = batch_target_representation,
                            mico_beta = self.mico_beta).unsqueeze(-1)
            else:
                raise ValueError("Invalid priority type")

            td_error_priority = (pred_val_index - target_value).pow(2)
            td_error_priority = td_error_priority.unsqueeze(-1)

            # TODO: Check why I need .unsqueeze(-1) here
            # NOTE: I prefer to define this in the main loop to don't store too much info in the tensordict
            # , but I am gonna leave it here
            # priority_tensor = (1 - self.priority_alpha) * td_error_priority + self.priority_alpha * mico_priority.unsqueeze(-1)

        if tensordict.device is not None:
            td_error_priority = td_error_priority.to(tensordict.device)
            mico_distance = mico_distance.to(tensordict.device)
            # priority_tensor = priority_tensor.to(tensordict.device)

        # tensordict.set(
        #     "total_priority",
        #     priority_tensor,
        #     inplace=True,
        # )

        tensordict.set(
            "td_error",
            td_error_priority,
            inplace=True,
        )

        tensordict.set(
            "mico_distance",
            mico_distance,
            inplace=True,
        )

        td_loss = distance_loss(pred_val_index, target_value, self.loss_function)
        td_loss = _reduce(td_loss, reduction=self.reduction)

        loss = ((1. - self.mico_weight) * td_loss + self.mico_weight * mico_loss)    

        # mico_loss = distance_loss(pred_dist, target_dist, self.loss_function)
        # mico_loss = _reduce(mico_loss, reduction=self.reduction)

        # loss = self.mu * mico_loss + (1 - self.mu) * td_loss

        td_out = TensorDict({"loss": loss, "td_loss": td_loss, "mico_loss": mico_loss}, [])
        # td_out = TensorDict({"loss": loss}, [])

        # TODO: Try frist the td loss alone
        return td_out
