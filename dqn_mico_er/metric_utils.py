import torch

EPSILON = 1e-9

def _sqrt(x):
  # zeros like instead of zeros
  # It is because vmap works with a weird way of broadcasting
  # and a weird structure based on tensors
  tol = torch.zeros_like(x)
  return torch.sqrt(torch.maximum(x, tol))


def cosine_distance(x, y):
  # NOTE: the cosine similarity is not calculate directly for 
  # instabilities observed when using `jnp.arccos`, but I'm using torch
  # so I don't know if I will need to do this
  numerator = torch.sum(x * y)
  denominator = torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))
  cos_similarity = numerator / (denominator + EPSILON)

  # cos_similarity = cos(theta)

  # NOTE: From, the Pythagorean trigometric identity
  # sin^2(theta) + cos^2(theta) = 1
  # you can get sin(theta) = sqrt(1 - cos^2(theta))
  # and the arctan2(sin(theta), cos(theta)) = theta
  return torch.arctan2(_sqrt(1. - cos_similarity**2), cos_similarity)


def squarify(x):
    # Squarify will take the input and adds a new dimension between the batch and the representation
    # so that the representation is repeated along the new dimension
    # To visualize thing of x as a matrix of batch_size x representation_dim
    # and squarify will place that matrix in a lateral way and repeat it along the new dimension j

    # NOTE: after squarify if you pick a i-th row all the elements (j-th index) in that row will be the same
    batch_size = x.shape[0]
    if len(x.shape) > 1:
        representation_dim = x.shape[-1]
        return x.tile((batch_size,)).view(batch_size, batch_size, representation_dim)
    return x.tile((batch_size,)).view(batch_size, batch_size)

def representation_distances(first_representations, second_representations, beta=0.1,
                             return_distance_components=False):
  """Compute distances between representations.
     In the paper, it corresponds to the calculation of the U term
     for each pair of representations in the batch (all-vs-all).

  This will compute the distances between two representations.

  Args:
    first_representations: first set of representations to use.
    second_representations: second set of representations to use.
    beta: float, weight given to cosine distance between representations.
    return_distance_components: bool, whether to return the components used for
      computing the distance.

  Returns:
    The distances between representations, combining the average of the norm of
    the representations and the distance given by the cosine similarity distance function.
  """
  batch_size = first_representations.shape[0]
  representation_dim = first_representations.shape[-1]

  # Squarify the representations and reshape them to make a pair-waise comparison with vmap
  first_squared_reps = squarify(first_representations)
  first_squared_reps = torch.reshape(first_squared_reps,
                                   [batch_size**2, representation_dim])
  
  # Squarify the representations and reshape them to make a pair-waise comparison with vmap
  # However, we now need to permute (transpose) the dimension 0, 1 to alternate the values
  # so that we have the pair-wise comparisons of all-vs-all
  second_squared_reps = squarify(second_representations)
  second_squared_reps = torch.permute(second_squared_reps, dims=(1, 0, 2))
  second_squared_reps = torch.reshape(second_squared_reps,
                                    [batch_size**2, representation_dim])
  
  # vmap will calculate the pairwise distance_fn along the dimension specified
  # in in_axes. In this case, will take the dim 0 of the first_squared_reps and
  # the dim 0 of the second_squared_reps and apply the distance
  # It vertorize the process of calculating the distance between all the pairs

  # NOTE: base distance corresponds to the second term in the U calculation in the paper
  # It calculates the angle between the representations in the paper
  # Check what function is using
  base_distances = torch.vmap(cosine_distance, in_dims=(0, 0))(first_squared_reps,
                                                         second_squared_reps)
  base_distances = base_distances
  # Sum along the second dimension and normalize the distance
  # NOTE: this is practically the first term of U in the paper
  norm_average = 0.5 * (torch.sum(torch.square(first_squared_reps), -1) +
                        torch.sum(torch.square(second_squared_reps), -1))
  
  if return_distance_components:
    return norm_average + beta * base_distances, norm_average, base_distances
  return norm_average + beta * base_distances

# NOTE: check in the main code if the output of this requires grad and the 
# other output must require grad
@torch.no_grad()
def target_distances(representations, rewards, cumulative_gamma):
  """Target distance using the metric operator. This is the T in the paper :D"""
  next_state_similarities = representation_distances(
      representations, representations)
  squared_rews = squarify(rewards).squeeze(-1)
  squared_rews_transp = squared_rews.T
  squared_rews = squared_rews.reshape((squared_rews.shape[0]**2))
  squared_rews_transp = squared_rews_transp.reshape(
      (squared_rews_transp.shape[0]**2))
  reward_diffs = torch.abs(squared_rews - squared_rews_transp)
  return reward_diffs + cumulative_gamma * next_state_similarities