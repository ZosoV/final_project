# Limit of resources
# 4 GPUS
# 1344 cores and 6TB of memory (RAM) per shared CPU QOS

# Check if my setting fill the requirements

num_games = 2
num_seeds = 3
num_cores = 32
num_ram = 6
num_algorithms = 4

total_cores = num_games * num_seeds * num_cores * num_algorithms
total_ram = num_games * num_seeds * num_cores * num_ram * num_algorithms

# convert ram that is in GB to TB
total_ram = total_ram / 1024

print(f"Total cores: {total_cores}")
print(f"Total RAM: {total_ram} TB")

if total_cores > 1344:
    print("Not enough cores")
else:
    print("Enough cores")

if total_ram > 6:
    print("Not enough RAM")
else:
    print("Enough RAM")