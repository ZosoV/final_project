# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN Agent with MICo loss."""

import collections
import functools
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from absl import logging


# from mico.atari import metric_utils
import metric_utils

NetworkType = collections.namedtuple('network', ['q_values', 'representation'])


@gin.configurable
class AtariDQNNetwork(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    representation = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(representation)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions,
                        kernel_init=initializer)(x)
    return NetworkType(q_values, representation)


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12, 14))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          mico_weight, distance_fn, loss_weights, bper_weight = 0):
  """Run the training step."""
  def loss_fn(params, bellman_target, target_r, target_next_r, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state)

    model_output = jax.vmap(q_online)(states)
    q_values = model_output.q_values
    q_values = jnp.squeeze(q_values)
    representations = model_output.representation
    representations = jnp.squeeze(representations)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    batch_bellman_loss = jax.vmap(losses.mse_loss)(bellman_target,
                                                      replay_chosen_q)
    bellman_loss = jnp.mean(loss_multipliers * batch_bellman_loss)
    online_dist = metric_utils.representation_distances(
        representations, target_r, distance_fn)
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    metric_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist,
                                                       target_dist))
    loss = ((1. - mico_weight) * bellman_loss +
            mico_weight * metric_loss)
    
    # Current vs Next Distance without squarify
    # NOTE: I could try to use online next representation instead of target_next_r
    # NOTE: when bper_weight we are using PER and we don't need to compute the experience distance
    if bper_weight > 0:
      experience_distances = metric_utils.current_next_distances(
        current_state_representations=representations,
        next_state_representations=target_next_r,
        distance_fn = distance_fn,)
    else:
      experience_distances = jnp.zeros_like(batch_bellman_loss)

    return jnp.mean(loss), (bellman_loss, metric_loss, batch_bellman_loss, experience_distances)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = target_outputs(
      q_target, states, next_states, rewards, terminals, cumulative_gamma)
  (loss, component_losses), grad = grad_fn(online_params, bellman_target,
                                           target_r, target_next_r, loss_weights)
  bellman_loss, metric_loss, batch_bellman_loss, experience_distances = component_losses
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, bellman_loss, metric_loss, batch_bellman_loss, experience_distances


def target_outputs(target_network, states, next_states, rewards, terminals,
                   cumulative_gamma):
  """Compute the target Q-value."""
  curr_state_representation = jax.vmap(target_network, in_axes=(0))(
      states).representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  next_state_output = jax.vmap(target_network, in_axes=(0))(next_states)
  next_state_q_vals = next_state_output.q_values
  next_state_q_vals = jnp.squeeze(next_state_q_vals)
  next_state_representation = next_state_output.representation
  next_state_representation = jnp.squeeze(next_state_representation)
  replay_next_qt_max = jnp.max(next_state_q_vals, 1)
  return (
      jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                            (1. - terminals)),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))


@gin.configurable
class MetricDQNBPERAgent(dqn_agent.JaxDQNAgent):
  """DQN Agent with the MICo loss."""

  def __init__(self, 
               num_actions, 
               summary_writer=None,
               mico_weight=0.01, 
               distance_fn=metric_utils.cosine_distance,
               replay_scheme='uniform',
               bper_weight=0, # PER: 0 and BPER: 1
               ):
    self._mico_weight = mico_weight
    self._distance_fn = distance_fn
    
    network = AtariDQNNetwork
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)
    
    self._replay_scheme = replay_scheme
    self._bper_weight = bper_weight
    # NOTE: As I create a function that only create the prioritized replay
    # I don't need to call it again here. It is is called above with the super
    # that calls the original DQN agent
    # self._replay = self._build_replay_buffer() 
    logging.info(
        'Creating %s agent with the following parameters:',
        self.__class__.__name__,
    )
    logging.info('\t mico_weight: %f', self._mico_weight)
    logging.info('\t distance_fn: %s', self._distance_fn)
    logging.info('\t replay_scheme: %s', self._replay_scheme)
    logging.info('\t bper_weight: %f', bper_weight)
    logging.info('\t gamma: %f', self.gamma)
    logging.info('\t update_horizon: %f', self.update_horizon)
    logging.info('\t min_replay_history: %d', self.min_replay_history)
    logging.info('\t update_period: %d', self.update_period)
    logging.info('\t target_update_period: %d', self.target_update_period)
    logging.info('\t epsilon_train: %f', self.epsilon_train)
    logging.info('\t epsilon_eval: %f', self.epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', self.epsilon_decay_period)
    logging.info('\t optimizer: %s', self.optimizer)
    logging.info('\t seed: %d', self._seed)
    logging.info('\t loss_type: %s', self._loss_type)
    logging.info('\t preprocess_fn: %s', self.preprocess_fn)
    logging.info('\t summary_writing_frequency: %d', self.summary_writing_frequency)
    logging.info('\t allow_partial_reload: %s', self.allow_partial_reload)
    
  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))

    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
    )

    sampling_distribution = samplers.PrioritizedSamplingDistribution(
            seed=self._seed
       )
    return replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=sampling_distribution,
    )

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          # Weight the loss by the inverse priorities.
          # NOTE: they don't divide to N (size of buffer) because many optimizer 
          # are scale invariant as Adam
          # NOTE: they use sqrt(prob) instead of prob because they practically
          # are setting beta = 0.5.
          # normaly the value should (1/P)^beta => 1/sqrt(P) considering beta = 0.5
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        (self.optimizer_state, self.online_params,
         loss, bellman_loss, metric_loss, 
         batch_bellman_loss, experience_distances) = train(
             self.network_def,
             self.online_params,
             self.target_network_params,
             self.optimizer,
             self.optimizer_state,
             self.replay_elements['state'],
             self.replay_elements['action'],
             self.replay_elements['next_state'],
             self.replay_elements['reward'],
             self.replay_elements['terminal'],
             self.cumulative_gamma,
             self._mico_weight,
             self._distance_fn,
             loss_weights,
             self._bper_weight)
        
        if self._replay_scheme == 'prioritized':
          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake
          # we leave it as is here, using the more direct sqrt(). Taking the
          # square root "makes sense", as we are dealing with a squared loss.
          # Add a small nonzero value to the loss to avoid 0 priority items.
          # While technically this may be okay, setting all items to 0
          # priority will cause troubles, and also result in 1.0 / 0.0 = NaN
          # correction terms.

          # NOTE: Option we can in the same way as the loss weights, use the sqrt of the
          # experience distance.
          # priorities = (1 - self._bper_weight) * jnp.sqrt(loss + 1e-10) + self._bper_weight * jnp.sqrt(experience_distances + 1e-10)
          priorities = (1 - self._bper_weight) * jnp.sqrt(batch_bellman_loss + 1e-10) + self._bper_weight * experience_distances

          self._replay.update(
              self.replay_elements['indices'],
              priorities=priorities,
          )

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
        #   summary = tf.compat.v1.Summary(value=[
        #       tf.compat.v1.Summary.Value(tag='Losses/Aggregate',
        #                                  simple_value=loss),
        #       tf.compat.v1.Summary.Value(tag='Losses/Bellman',
        #                                  simple_value=bellman_loss),
        #       tf.compat.v1.Summary.Value(tag='Losses/Metric',
        #                                  simple_value=metric_loss),
        #   ])
        #   self.summary_writer.add_summary(summary, self.training_steps)
            with self.summary_writer.as_default():
                tf.summary.scalar('Losses/Aggregate', loss, step=self.training_steps)
                tf.summary.scalar('Losses/Bellman', bellman_loss, step=self.training_steps)
                tf.summary.scalar('Losses/Metric', metric_loss, step=self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1