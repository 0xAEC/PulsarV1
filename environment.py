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

    def __init__(self, size=10, n_obstacles=0, render_mode='human', add_door_key=True):
        super().__init__()
        self.size = size
        self.n_obstacles = n_obstacles
        self.add_door_key = add_door_key
        
        self.window_size = 512
        self._agent_location = None
        self._target_location = None
        self.obstacles = []
        self._door_location = None
        self._key_location = None
        self.has_key = False
        self._door_open = False

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "obstacles": gym.spaces.Sequence(gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)),
            # NEW OBSERVATION SPACE COMPONENTS
            "door_loc": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "key_loc": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "has_key": gym.spaces.Discrete(2), # 0 for False, 1 for True
            "door_open": gym.spaces.Discrete(2)
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {0: np.array([1, 0]), 1: np.array([0, -1]), 2: np.array([-1, 0]), 3: np.array([0, 1])}
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        # Use a dummy location off-screen if an object doesn't exist
        dummy_loc = np.array([-1, -1]) 
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "obstacles": self.obstacles,
            "door_loc": self._door_location if self._door_location is not None else dummy_loc,
            "key_loc": self._key_location if self._key_location is not None else dummy_loc,
            "has_key": 1 if self.has_key else 0,
            "door_open": 1 if self._door_open else 0
        }

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Agent starts on the left side
        self._agent_location = np.array([1, self.np_random.integers(1, self.size - 1)])
        # Target starts on the right side
        self._target_location = np.array([self.size - 2, self.np_random.integers(1, self.size - 1)])

        occupied_locations = {tuple(self._agent_location), tuple(self._target_location)}
        
        # Reset door/key state
        self.has_key = False
        self._door_open = False
        self._door_location = None
        self._key_location = None

        # Build a wall down the middle to force use of the door
        wall_x = self.size // 2
        self.obstacles = [np.array([wall_x, y]) for y in range(self.size)]
        
        if self.add_door_key:
            # Create a single door opening in the wall
            door_y = self.np_random.integers(1, self.size - 1)
            self._door_location = np.array([wall_x, door_y])
            # THIS IS THE CORRECTED LINE:
            self.obstacles = [obs for obs in self.obstacles if not np.array_equal(obs, self._door_location)]
            
            # Place the key somewhere on the agent's side of the wall
            key_x = self.np_random.integers(0, wall_x)
            key_y = self.np_random.integers(0, self.size)
            self._key_location = np.array([key_x, key_y])
            occupied_locations.add(tuple(self._key_location))

        # Add other random obstacles, avoiding the key and door path
        for _ in range(self.n_obstacles):
            loc = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            # Avoid placing on the wall or too close to important items
            if loc[0] != wall_x and loc not in occupied_locations:
                 self.obstacles.append(np.array(loc))
                 occupied_locations.add(loc)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human": self._render_frame()
        return observation, info
    
    def _create_impassable_maze(self):
        """
        Helper function to make the current target unreachable by walling it off.
        This forces the agent's A* planner to fail, testing its "creative" faculties.
        """
        tx, ty = self._target_location
        # Add walls around the target location, leaving no gaps.
        # Ensure we don't place obstacles on the agent's or target's starting squares.
        occupied = {tuple(self._agent_location), tuple(self._target_location)}
        
        # Build a 3x3 box of obstacles around the target
        new_obstacles_to_add = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue # Don't place an obstacle ON the target
                
                new_obstacle_loc = (tx + i, ty + j)
                
                # Check if the location is valid and not already occupied by the agent/target
                if (0 <= new_obstacle_loc[0] < self.size and 
                    0 <= new_obstacle_loc[1] < self.size and
                    new_obstacle_loc not in occupied):
                    
                    # Also, ensure we don't accidentally wall the agent in.
                    if abs(new_obstacle_loc[0] - self._agent_location[0]) + abs(new_obstacle_loc[1] - self._agent_location[1]) > 1:
                        new_obstacles_to_add.append(np.array(new_obstacle_loc))

        self.obstacles.extend(new_obstacles_to_add)
        # Remove duplicates that might have been added
        self.obstacles = list(set(map(tuple, self.obstacles)))
        self.obstacles = [np.array(obs) for obs in self.obstacles]

    def step(self, action):
        direction = self._action_to_direction[action]
        proposed_location = self._agent_location + direction

        # Boundary checks
        if not (0 <= proposed_location[0] < self.size and 0 <= proposed_location[1] < self.size):
            proposed_location = self._agent_location

        # Obstacle collision checks
        if any(np.array_equal(proposed_location, obs) for obs in self.obstacles):
            proposed_location = self._agent_location # Blocked

        # Door interaction logic
        if self._door_location is not None and np.array_equal(proposed_location, self._door_location) and not self._door_open:
            if self.has_key:
                self._door_open = True
                print("Door Unlocked!")
                # Reward for using the key
                reward_modifier = 0.5
            else:
                proposed_location = self._agent_location # Blocked by door
        else:
            reward_modifier = 0

        self._agent_location = proposed_location

        # Key pickup logic
        if self._key_location is not None and np.array_equal(self._agent_location, self._key_location):
            self.has_key = True
            self._key_location = None # Key is "removed"
            print("Key Obtained!")
            # Reward for picking up the key
            reward_modifier = 0.2

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if terminated else (-0.02 + reward_modifier)  # Smaller step penalty

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human": self._render_frame()
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

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(obstacle[0] * pix_square_size, obstacle[1] * pix_square_size, pix_square_size, pix_square_size))

        # Draw Key
        if self._key_location is not None:
            pygame.draw.rect(canvas, (255, 215, 0), pygame.Rect(self._key_location[0] * pix_square_size, self._key_location[1] * pix_square_size, pix_square_size, pix_square_size))
            key_center = (self._key_location + 0.5) * pix_square_size
            pygame.draw.circle(canvas, (180, 150, 0), key_center, pix_square_size / 4, 3)

        # Draw Door
        if self._door_location is not None:
            door_color = (0, 255, 0) if self._door_open else (139, 69, 19) # Green if open, brown if closed
            pygame.draw.rect(canvas, door_color, pygame.Rect(self._door_location[0] * pix_square_size, self._door_location[1] * pix_square_size, pix_square_size, pix_square_size))

        # Draw the target
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(self._target_location[0] * pix_square_size, self._target_location[1] * pix_square_size, pix_square_size, pix_square_size))
        
        # Draw the agent
        agent_center = (self._agent_location + 0.5) * pix_square_size
        pygame.draw.circle(canvas, (0, 0, 255), agent_center, pix_square_size / 3)
        if self.has_key: # Draw a smaller circle inside the agent if it has the key
             pygame.draw.circle(canvas, (255, 215, 0), agent_center, pix_square_size / 6)

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (0, x * pix_square_size), (self.window_size, x * pix_square_size), width=1)
            pygame.draw.line(canvas, (200, 200, 200), (x * pix_square_size, 0), (x * pix_square_size, self.window_size), width=1)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

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
        agent_loc = tuple(int(c) for c in obs["agent"])
        target_loc = tuple(int(c) for c in obs["target"])
        door_loc = tuple(int(c) for c in obs["door_loc"])
        key_loc_raw = tuple(int(c) for c in obs["key_loc"])
        has_key = bool(obs["has_key"])
        door_open = bool(obs["door_open"])

        # Determine qualitative state ID
        if agent_loc == target_loc:
            state_id = "TASK_COMPLETE"
        elif has_key:
            state_id = "HAS_KEY_DOOR_AHEAD"
        elif key_loc_raw[0] != -1: # Key is visible
             state_id = "KEY_VISIBLE_DOOR_LOCKED"
        else: # Should not happen in normal scenarios but is a safe fallback
            state_id = "STATE_UNKNOWN"
        
        key_loc = key_loc_raw if key_loc_raw[0] != -1 else None

        properties = {
            'agent_loc': agent_loc,
            'target_loc': target_loc,
            'obstacles': frozenset(tuple(int(c) for c in o) for o in obs.get('obstacles', [])),
            'door_loc': door_loc if door_loc[0] != -1 else None,
            'key_loc': key_loc,
            'agent_has_key': has_key,
            'door_open': door_open,
            'grid_size': self.grid_size
        }
        
        # Cache based on the full property set for uniqueness
        cache_key = (state_id, tuple(sorted(properties.items())))

        if cache_key not in self.state_cache:
            self.state_cache[cache_key] = StateHandle(id=state_id, properties=properties)
            
        return self.state_cache[cache_key]
