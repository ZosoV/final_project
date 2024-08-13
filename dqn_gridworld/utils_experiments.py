import numpy
import torch

# Create a matrix of all possible states taking into account that
# state is given by position of agent
grid_size = 5

visiting_count = torch.zeros((grid_size, grid_size))

state_positions = data['observation']

# Counting new visits
visiting_count[state_positions[:, 0], state_positions[:, 1]] += 1

# TODO: Save the visiting count at the end in a file
# Save the visiting count
torch.save(visiting_count, 'visiting_count.pt')

# Define a function to calculate the visiting frequency and get the entropy
def get_entropy(visiting_count):
    visiting_frequency = visiting_count / visiting_count.sum()
    entropy = - (visiting_frequency * torch.log(visiting_frequency)).sum()
    return entropy

# Calculate the entropy
entropy = get_entropy(visiting_count)