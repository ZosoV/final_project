# Deep Bisimulation for Control (DBC)

- DBC aims to learn a representation of the environment's states where distances between states in the learned representation space reflect their bisimulation distances.
    - How that works? I think is similar to contrastive learning?

- Loss Function: The loss function used in DBC typically involves terms that penalize discrepancies between the bisimulation metric computed in the latent space and the actual rewards and transitions observed. This may include mean squared error (MSE) between predicted and actual rewards and between predicted next-state representations.

- The bisimulation metrics used are softened through a **pseudometric space**, allowing for the measurement of "behavioral similarity" between states.