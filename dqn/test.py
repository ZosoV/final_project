import hydra
from utils_cartpole import print_hyperparameters

@hydra.main(config_path=".", config_name="config_cartpole", version_base=None)
def main(cfg: "DictConfig"):

    print_hyperparameters(cfg)

if __name__ == "__main__":
    main()
