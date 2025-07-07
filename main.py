# main.py

"""
This file is the new, sole entry point for execution. It assembles the engine 
components from the other modules and the universe definition to run a simulation.
"""

import numpy as np
import random
import time
import copy
import pygame

# New modular imports
from environment import SimpleGridworld, PerceptionSystem
from cognitive_engine import SimplifiedOrchOREmulator
from core_abstractions import GoalState, StateHandle
from universe_definitions import TWO_QUBIT_UNIVERSE_CONFIG

# CORRECTED IMPORTS: Add ALL necessary default configs for the unpacking pattern
from configurations import (
    DEFAULT_COGNITIVE_FIREWALL_CONFIG,
    DEFAULT_SMN_CONFIG,
    DEFAULT_LOT_CONFIG
)


# ---------------------------------------------------------------------------
# Main Execution Block (Iota Directive: Embodied Agent Simulation)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    seed_val = int(time.time())
    np.random.seed(seed_val)
    random.seed(seed_val)
    print(f"Global random seed set to: {seed_val}")

    MASTER_VERBOSE_LEVEL = 1 # 0: Quiet, 1: Standard, 2: Detailed
    
    # 1. Initialize the World and the Agent's "Senses"
    print("--- Initializing Environment and Perception System ---")
    GRID_SIZE = 15
    N_OBSTACLES = 12
    env = SimpleGridworld(size=GRID_SIZE, n_obstacles=N_OBSTACLES, render_mode="human")
    perception_system = PerceptionSystem(grid_size=GRID_SIZE)

    # 2. Configure the Agent's "Mind"
    agent_config = {
        'universe': TWO_QUBIT_UNIVERSE_CONFIG,
        'agent_id': "pathfinder_agent_007",
        'verbose': MASTER_VERBOSE_LEVEL,
        'working_memory_max_depth': 10,
        'cognitive_firewall_config': {**DEFAULT_COGNITIVE_FIREWALL_CONFIG, 'enabled': True},
        'smn_general_config': {**DEFAULT_SMN_CONFIG, 'enabled': True, 'enable_influence_matrix': False},
        'lot_config': {**DEFAULT_LOT_CONFIG, 'enabled': True, 'log_level_details': {
            'system.embodied_cycle_start': True,
            'arbiter.choice': True,
            'motor.plan_astar_start': True,
            'motor.plan_astar_success': True,
            'motor.plan_astar_fail': True,
            'feedback.ltm_store_plan': True
        }}
    }
    
    # 3. Instantiate the Agent
    print("\n--- Initializing Embodied Cognitive Agent ---")
    agent = SimplifiedOrchOREmulator(
        **agent_config,
        action_space=env.action_space,
        perception_system=perception_system
    )
    
    # 4. Run the Main Simulation Loop
    # 4. Run the Main Simulation Loop
    print("\n--- Starting Main Simulation Loop ---")
    num_episodes = 5
    for i_episode in range(num_episodes):
        print(f"\n======== Starting Episode {i_episode + 1}/{num_episodes} ========")
        raw_observation, info = env.reset()

        # Set this episode to be impossible if it's even-numbered
        is_impossible = (i_episode + 1) % 2 == 0
        if is_impossible:
            print("!!! This is an IMPOSSIBLE episode. Testing agent's response to failure. !!!")
            env._create_impassable_maze()
            # CRITICAL: We must get the observation again after modifying the environment
            raw_observation = env._get_obs()
        
        # PERCEIVE THE FINAL STATE OF THE WORLD **BEFORE** SETTING THE GOAL
        initial_perception = perception_system.observe_to_state_handle(raw_observation)
        goal_target_state_props = initial_perception.properties.get('target_loc') # Use correct property key 'target_loc'

        # Create a clear goal description
        goal_description = f"Reach target at {goal_target_state_props} (Door/Key Puzzle)"
        
        # The StateHandle represents the terminal condition for success
        final_state_condition = StateHandle(id="TASK_COMPLETE", properties={'target': goal_target_state_props})

        agent_goal = GoalState(
            current_goal=goal_description,
            steps=[{"name": "Solve the puzzle and reach the target", "target_state": final_state_condition}]
        )
        agent.set_goal_state(agent_goal)
        print(f"Agent assigned goal: {agent.current_goal_state_obj}")
        
        terminated = False
        truncated = False
        reward = 0.0 # Initial reward is 0

        # The agent's "life" loop
        for t in range(150): # Max steps per episode
            if terminated or truncated:
                break
                
            env.render()
            
            # Agent's core cognitive-motor loop. It needs to know if last state was terminal.
            # On the first step, terminated is False.
            chosen_action = agent.run_cycle(raw_observation, reward, terminated)
            
            if chosen_action is not None:
                raw_observation, reward, terminated, truncated, info = env.step(chosen_action)
            else: # Agent is thinking, world is static
                reward = 0.0
                terminated = False 

            if terminated:
                print(f"Episode {i_episode+1} finished after {t+1} timesteps. Goal Reached!")
        
        # Allow agent to process the final, successful state.
        if terminated:
             if agent.verbose >= 1: print("  Running final cognitive cycle to process success...")
             final_reward = 1.0 
             # The run_cycle after termination is crucial for learning from the success.
             agent.run_cycle(raw_observation, final_reward, terminated=True)
        
        agent.print_internal_state_summary(indent="  ")
        time.sleep(1)

    env.close()
    print("\n--- All Episodes Completed ---")
