# FILE: environment.py
# PURPOSE: Defines the agent's world and physical body.

import pygame
import gymnasium as gym
import numpy as np
from typing import Tuple, List

from core_abstractions import StateHandle

# =========================================================================
# Class 1: The World Simulator
# =========================================================================
class SimpleGridworld(gym.Env):
    """
    A simple 2D Grid World environment. The agent's "body" is a single point.
    The world contains a goal location and obstacles.
    This class adheres to the Gymnasium API.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, size=10, render_mode='human'):
        super().__init__()
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),   # right (East)
            1: np.array([0, -1]),  # up (North) - Note: In PyGame, Y is inverted. Correcting here for standard coordinates.
            2: np.array([-1, 0]),  # left (West)
            3: np.array([0, 1]),   # down (South)
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if terminated else -0.1  # Reward for reaching the target, small penalty for each step
        
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
        pix_square_size = self.window_size / self.size

        # Draw the target
        pygame.draw.rect(
            canvas, (255, 0, 0),
            pygame.Rect(
                self._target_location[0] * pix_square_size,
                self._target_location[1] * pix_square_size,
                pix_square_size, pix_square_size,
            ),
        )
        # Draw the agent
        pygame.draw.circle(
            canvas, (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, 0,
                (0, x * pix_square_size),
                (self.window_size, x * pix_square_size),
                width=3
            )
            pygame.draw.line(
                canvas, 0,
                (x * pix_square_size, 0),
                (x * pix_square_size, self.window_size),
                width=3
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# =========================================================================
# Class 2: The "Perception System"
# This is a simple version for now. Later, this would be a powerful VAE or Transformer.
# =========================================================================
class PerceptionSystem:
    """
    Translates raw environment observations into abstract StateHandles.
    """
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        # For now, our "concepts" are just relative positions.
        self.state_cache = {}

    def observe_to_state_handle(self, obs: dict) -> StateHandle:
        """
        Converts a gridworld observation dictionary into a single StateHandle.
        
        The StateHandle's ID will be a qualitative description of the target's
        position relative to the agent.
        """
        agent_loc = obs["agent"]
        target_loc = obs["target"]
        
        delta = target_loc - agent_loc
        
        # Determine the primary and secondary direction
        if abs(delta[0]) > abs(delta[1]):
            # More horizontal than vertical
            primary_dir = "EAST" if delta[0] > 0 else "WEST"
            if delta[1] > 0: secondary_dir = "SOUTH"
            elif delta[1] < 0: secondary_dir = "NORTH"
            else: secondary_dir = ""
        elif abs(delta[1]) > abs(delta[0]):
            # More vertical than horizontal
            primary_dir = "SOUTH" if delta[1] > 0 else "NORTH"
            if delta[0] > 0: secondary_dir = "EAST"
            elif delta[0] < 0: secondary_dir = "WEST"
            else: secondary_dir = ""
        elif delta[0] == 0 and delta[1] == 0:
            return StateHandle(id="TARGET_REACHED")
        else: # Perfectly diagonal
            primary_dir = "SOUTH" if delta[1] > 0 else "NORTH"
            secondary_dir = "EAST" if delta[0] > 0 else "WEST"
        
        # Create a qualitative ID
        state_id = f"TARGET_{primary_dir}"
        if secondary_dir:
            state_id += f"_{secondary_dir}"
        
        # Use caching to return the same object for the same ID
        if state_id not in self.state_cache:
            self.state_cache[state_id] = StateHandle(id=state_id)
            
        return self.state_cache[state_id]
