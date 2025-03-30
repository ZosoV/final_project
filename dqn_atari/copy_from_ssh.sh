# Provide the source path as the first argument and the destination folder as the second argument
# Default destination folder is "outputs/" if not provided
destination_folder=${2:-outputs/}

# Check if the destination folder exists
if [ ! -d "$destination_folder" ]; then
    echo "Destination folder $destination_folder does not exist. Creating it now."
    mkdir -p "$destination_folder"
else
    echo "Destination folder $destination_folder already exists."
fi

# Copy the folder from the remote server to the specified destination
scp -r guarniov@bluebear.bham.ac.uk:$1 "$destination_folder"

# Extract the experiment name from the provided path
experiment_name=$(basename "$1" | awk -F'/' '{print $1}')
echo "Experiment Name from ssh: $experiment_name"

new_experiment_name=$(echo "$1" | cut -d'/' -f9)
echo "Experiment New name: $new_experiment_name"

# Chaging the name of the copied folder
if [ -d "$destination_folder/$experiment_name" ]; then
    mv "$destination_folder/$experiment_name" "$destination_folder/$new_experiment_name"
    echo "Renamed $destination_folder/$experiment_name to $destination_folder/$new_experiment_name"
else
    echo "Directory $destination_folder/$experiment_name does not exist."
fi

# Conda activate final-project
# Activate the conda environment
# conda activate final-project

# Syncronize with wandb the folder
echo "Syncing with wandb..."
wandb sync $destination_folder/$new_experiment_name
echo "Sync completed."