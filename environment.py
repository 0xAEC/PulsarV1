
# PURPOSE: Defines the agent's world and physical body.

import pygame
import gymnasium as gym
import numpy as np
from typing import Tuple, List

from core_abstractions import StateHandle

class SimpleGridworld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, size=15, n_obstacles=12, render_mode='human', add_door_key=True):
        super().__init__()
        self.size = size
        self.n_obstacles = n_obstacles
        self.add_door_key = add_door_key
        self.window_size = 512
        
        # --- NEW DYNAMICS ---
        self.max_energy = 100.0
        self.agent_energy = self.max_energy
        self._charging_pad_location = np.array([1, 1])
        self._bonus_target_location = None
        self.bonus_target_timer = 0
        self.bonus_target_lifespan = 50 # steps
        # --- END NEW DYNAMICS ---
        
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
            "door_loc": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "key_loc": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "has_key": gym.spaces.Discrete(2),
            "door_open": gym.spaces.Discrete(2),
            # --- NEW OBSERVATIONS ---
            "agent_energy": gym.spaces.Box(0, self.max_energy, shape=(1,), dtype=float),
            "charging_pad_loc": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "bonus_target_loc": gym.spaces.Box(-1, size - 1, shape=(2,), dtype=int), # -1 indicates inactive
            "bonus_target_timer": gym.spaces.Box(0, self.bonus_target_lifespan, shape=(1,), dtype=int),
        })
        self.action_space = gym.spaces.Discrete(4) # 0: down, 1: left, 2: up, 3: right
        self._action_to_direction = {0: np.array([1, 0]), 1: np.array([0, -1]), 2: np.array([-1, 0]), 3: np.array([0, 1])}
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        dummy_loc = np.array([-1, -1])
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "obstacles": self.obstacles,
            "door_loc": self._door_location if self._door_location is not None else dummy_loc,
            "key_loc": self._key_location if self._key_location is not None else dummy_loc,
            "has_key": 1 if self.has_key else 0,
            "door_open": 1 if self._door_open else 0,
            # --- NEW OBSERVATIONS ---
            "agent_energy": np.array([self.agent_energy], dtype=float),
            "charging_pad_loc": self._charging_pad_location,
            "bonus_target_loc": self._bonus_target_location if self._bonus_target_location is not None else dummy_loc,
            "bonus_target_timer": np.array([self.bonus_target_timer], dtype=int),
        }

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_energy = self.max_energy
        self._bonus_target_location = None
        self.bonus_target_timer = 0
        
        occupied_locations = {tuple(self._charging_pad_location)}
        
        # Agent and Target location
        self._agent_location = np.array([self.np_random.integers(1, self.size - 1), self.np_random.integers(1, self.size-1)])
        occupied_locations.add(tuple(self._agent_location))
        self._target_location = np.array([self.size - 2, self.np_random.integers(1, self.size - 1)])
        while tuple(self._target_location) in occupied_locations:
            self._target_location = np.array([self.size - 2, self.np_random.integers(1, self.size - 1)])
        occupied_locations.add(tuple(self._target_location))

        self.has_key = False
        self._door_open = False
        self._door_location = None
        self._key_location = None
        
        # Wall and Door
        wall_x = self.size // 2
        self.obstacles = [np.array([wall_x, y]) for y in range(self.size)]
        if self.add_door_key:
            door_y = self.np_random.integers(1, self.size - 1)
            self._door_location = np.array([wall_x, door_y])
            self.obstacles = [obs for obs in self.obstacles if not np.array_equal(obs, self._door_location)]
            key_x = self.np_random.integers(0, wall_x)
            key_y = self.np_random.integers(0, self.size)
            self._key_location = np.array([key_x, key_y])
            while tuple(self._key_location) in occupied_locations:
                 self._key_location = np.array([self.np_random.integers(0, wall_x), self.np_random.integers(0, self.size)])
            occupied_locations.add(tuple(self._key_location))
            
        for _ in range(self.n_obstacles):
            loc = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            if loc[0] != wall_x and loc not in occupied_locations:
                self.obstacles.append(np.array(loc)); occupied_locations.add(loc)
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        # --- Energy and Time Decay ---
        self.agent_energy = max(0.0, self.agent_energy - 0.5)
        if self.bonus_target_timer > 0:
            self.bonus_target_timer -= 1
            if self.bonus_target_timer == 0:
                self._bonus_target_location = None # Despawn
                print("Bonus target despawned!")

        # --- Spawn Bonus Target ---
        if self._bonus_target_location is None and self.np_random.random() < 0.02: # 2% chance per step
             self._bonus_target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
             self.bonus_target_timer = self.bonus_target_lifespan
             print(f"Bonus target spawned at {self._bonus_target_location} for {self.bonus_target_timer} steps!")

        direction = self._action_to_direction[action]
        proposed_location = self._agent_location + direction
        
        info = self._get_info()
        info['interacted_with'] = None 
        info['preconditions'] = {'has_key': self.has_key}

        reward_modifier = 0.0
        
        if not (0 <= proposed_location[0] < self.size and 0 <= proposed_location[1] < self.size):
            proposed_location = self._agent_location # Bump into wall
        if any(np.array_equal(proposed_location, obs) for obs in self.obstacles):
            info['interacted_with'] = 'wall'
            proposed_location = self._agent_location

        if self._door_location is not None and np.array_equal(proposed_location, self._door_location) and not self._door_open:
            info['interacted_with'] = 'door'
            if self.has_key: 
                self._door_open = True
                print("Door Unlocked!")
                reward_modifier += 0.5
            else:
                proposed_location = self._agent_location

        self._agent_location = proposed_location
        
        # --- New Interaction Logic ---
        if np.array_equal(self._agent_location, self._charging_pad_location):
             self.agent_energy = min(self.max_energy, self.agent_energy + 5.0)
             info['interacted_with'] = 'charging_pad'
        
        if self._key_location is not None and np.array_equal(self._agent_location, self._key_location):
            self.has_key = True
            self._key_location = None
            info['interacted_with'] = 'key'
            print("Key Obtained!")
            reward_modifier += 0.2

        if self._bonus_target_location is not None and np.array_equal(self._agent_location, self._bonus_target_location):
            info['interacted_with'] = 'bonus_target'
            reward_modifier += 0.8
            self._bonus_target_location = None
            self.bonus_target_timer = 0
            print("Bonus target collected!")

        terminated_by_goal = np.array_equal(self._agent_location, self._target_location)
        if terminated_by_goal:
            info['interacted_with'] = 'target'
        
        # --- Energy Death ---
        terminated_by_death = self.agent_energy <= 0
        if terminated_by_death:
            print("Agent ran out of energy!")

        terminated = terminated_by_goal or terminated_by_death

        if terminated_by_goal:
            reward = 1.0 + reward_modifier
        elif terminated_by_death:
            reward = -2.0 # Large penalty for dying
        else:
            reward = -0.02 + reward_modifier # Small time penalty
        
        return self._get_obs(), reward, terminated, False, info

    def render(self): return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size + 200, self.window_size)) # Widen for UI
            pygame.display.set_caption("Directive Rho Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw charging pad
        pygame.draw.rect(canvas, (200, 255, 200), pygame.Rect(self._charging_pad_location[1]*pix_square_size, self._charging_pad_location[0]*pix_square_size, pix_square_size, pix_square_size))
        # Draw bonus target
        if self._bonus_target_location is not None:
             pygame.draw.rect(canvas, (255, 165, 0), pygame.Rect(self._bonus_target_location[1]*pix_square_size, self._bonus_target_location[0]*pix_square_size, pix_square_size, pix_square_size))

        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, (0,0,0), pygame.Rect(obstacle[1]*pix_square_size, obstacle[0]*pix_square_size, pix_square_size, pix_square_size))
        if self._key_location is not None:
            pygame.draw.rect(canvas, (255,215,0), pygame.Rect(self._key_location[1]*pix_square_size, self._key_location[0]*pix_square_size, pix_square_size, pix_square_size))
        if self._door_location is not None:
            pygame.draw.rect(canvas, (0,255,0) if self._door_open else (139,69,19), pygame.Rect(self._door_location[1]*pix_square_size, self._door_location[0]*pix_square_size, pix_square_size, pix_square_size))
        
        pygame.draw.rect(canvas, (255,0,0), pygame.Rect(self._target_location[1]*pix_square_size, self._target_location[0]*pix_square_size, pix_square_size, pix_square_size))
        
        agent_center = (self._agent_location[1]+0.5)*pix_square_size, (self._agent_location[0]+0.5)*pix_square_size
        pygame.draw.circle(canvas, (0,0,255), agent_center, pix_square_size/3)
        if self.has_key:
            pygame.draw.circle(canvas, (255,215,0), agent_center, pix_square_size/6)

        for x in range(self.size+1):
            pygame.draw.line(canvas,(200,200,200),(0,x*pix_square_size),(self.window_size,x*pix_square_size),width=1)
            pygame.draw.line(canvas,(200,200,200),(x*pix_square_size,0),(x*pix_square_size,self.window_size),width=1)
        
        pixel_data = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))
        
        if self.render_mode=="human":
            # --- Draw UI Panel ---
            ui_panel = pygame.Surface((200, self.window_size))
            ui_panel.fill((50, 50, 50))

            # Energy bar
            energy_text = self.font.render(f"Energy: {self.agent_energy:.1f}", True, (255,255,255))
            ui_panel.blit(energy_text, (10, 10))
            energy_ratio = self.agent_energy / self.max_energy
            bar_color = (int(255 * (1-energy_ratio)), int(255 * energy_ratio), 0)
            pygame.draw.rect(ui_panel, (80,80,80), [10, 40, 180, 20])
            if energy_ratio > 0:
                pygame.draw.rect(ui_panel, bar_color, [10, 40, 180 * energy_ratio, 20])
            
            # Bonus timer
            if self.bonus_target_timer > 0:
                 bonus_text = self.font.render(f"Bonus Timer: {self.bonus_target_timer}", True, (255, 165, 0))
                 ui_panel.blit(bonus_text, (10, 80))
            
            self.window.blit(canvas, (0,0))
            self.window.blit(ui_panel, (self.window_size, 0))
            
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        return pixel_data
    
    def close(self):
        if self.window is not None: pygame.display.quit(); pygame.quit()
class PerceptionSystem:
    def __init__(self,grid_size:int): pass
    def observe_to_state_handle(self,obs:dict): return None
