import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import json


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_file = None, bisimulation_distance_file = None):

        # Load from a file
        if grid_file is None:
            raise ValueError("You must provide a grid file. As an example, you can use the grid.txt file.")
        
        self._grid = self.load_grid_from_file(grid_file)

        self.size = self._grid.shape[0] # if self._grid else size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.bisimulation_distance = None
        # If bisimulation_distance_file is provided, load it
        if bisimulation_distance_file is not None:
            # self.bisimulation_distance = np.array([line.split() for line in bisimulation_distance_file], dtype=np.float32)
            # Load JSON file
            with open(bisimulation_distance_file, 'r') as json_file:
                self.bisimulation_distance = json.load(json_file)
        # Set walls and target
        self._set_components()

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box( low = 0, high = self.size - 1, shape=(2,), dtype=np.int64),
                "behavioral_distance": spaces.Box( low = 0, high = np.inf, shape=(4,), dtype=np.float64),
            }
        )
        # self.observation_space = spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.int64)


        # We have 4 actions, corresponding to "down", "right", "up" and "left".
        # Action 0: Down
        # Action 1: Right
        # Action 2: Up
        # Action 3: Left
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def load_grid_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            self.height = len(lines)
            self.width = len(lines[0].split())
            self.grid_size = self.height

            if self.height != self.width:
                # Raise a error message
                raise ValueError("Invalid shape of widht and height must be equal, add extra walls to make it square")

            return np.array([line.split() for line in lines])
        
    def _set_components(self):
        self._walls_location = []
        self._targets_location = []
        for i in range(self.size):
            for j in range(self.size):
                if self._grid[i, j] == 'x':
                    self._walls_location.append((i, j))
                elif self._grid[i, j] == 'G':
                    self._targets_location.append((i, j))
        self._walls_location = set(self._walls_location)
        self._targets_location = set(self._targets_location)

    def _get_obs(self):
        behavioral_distance = np.zeros(len(self._action_to_direction))
        if self.bisimulation_distance is not None:
            behavioral_distance =  np.array(self.bisimulation_distance[str(tuple(self._agent_location))])
        return {"observation": self._agent_location, "behavioral_distance": behavioral_distance}

        # return self._agent_location

    def _get_info(self):
        # return {
        #     "distance": np.linalg.norm(
        #         self._agent_location - self._target_location, ord=1
        #     )
        # }
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # We will sample the agent's location randomly until it does not coincide with the target or walls's location
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        while tuple(self._agent_location) in self._walls_location or tuple(self._agent_location) in self._targets_location:
            self._agent_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # while np.array_equal(self._target_location, self._agent_location) or tuple(self._agent_location) in self._walls_location:
        #     self._agent_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # if to make sure we don't walk into a wall
        if tuple(self._agent_location + direction) not in self._walls_location:
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )


        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        terminated = tuple(self._agent_location) in self._targets_location

        # REWARD: The rewards is given by the minimum number of steps you should
        # need to take in a environment without obstacles
        # NOTE: Check it better
        reward = self.size * 2 if terminated else -1  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the walls
        for wall_location in self._walls_location:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(wall_location)[::-1], # I have to switch to display in order (rows, cols)
                    (pix_square_size, pix_square_size),
                ),
            )

        # Second we draw the targets
        for target_location in self._targets_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * np.array(target_location)[::-1], # I have to switch to display in order (rows, cols)
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    from gymnasium.envs.registration import register

    # Don't forget to register the environment
    register(
        id='GridWorldEnv-v0',
        entry_point='__main__:GridWorldEnv',
        max_episode_steps=200
    )

    from torchrl.envs import GymWrapper

    env = gym.make("GridWorldEnv-v0", render_mode = "rgb_array", grid_file='grid_envs/grid_world2.txt')
    env = GymWrapper(env, from_pixels=True, pixels_only=False, device="cpu")
    data = env.rollout(10)
    print(data)