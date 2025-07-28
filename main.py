# main.py

import os
import numpy as np
import random
import time
import copy
import pygame
import gymnasium as gym

# --- The Cognitive Architecture ---
from cognitive_engine import SimplifiedOrchOREmulator
from core_abstractions import GoalState

# --- Perception & World Model ---
from configurations import DEFAULT_VAE_CONFIG, DEFAULT_WORLD_MODEL_CONFIG, DEFAULT_LIFELONG_LEARNING_CONFIG
from train_perception import collect_experience_data, train_visual_cortex, train_world_model

# --- The Environment ---
from environment import SimpleGridworld

# ---------------------------------------------------------------------------
# Main Execution Block for Embodied AGI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Environment setup for cross-platform compatibility
    if 'XDG_RUNTIME_DIR' not in os.environ:
        dummy_xdg_dir = '/tmp/xdg_runtime_dir_fallback'
        os.makedirs(dummy_xdg_dir, exist_ok=True)
        os.chmod(dummy_xdg_dir, 0o700)
        os.environ['XDG_RUNTIME_DIR'] = dummy_xdg_dir

    # Set random seeds for reproducibility
    seed_val = int(time.time())
    np.random.seed(seed_val)
    random.seed(seed_val)
    print(f"Global random seed set to: {seed_val}")

    # --- Step 1: Bootstrap the Agent's Perception ---
    # This step is only run if the weight files don't exist.
    print("\n--- Bootstrapping Agent's Perception & World Model ---")
    bootstrap_env = SimpleGridworld(render_mode='rgb_array')
    
    vae_path = DEFAULT_VAE_CONFIG['MODEL_PATH']
    wm_path = DEFAULT_WORLD_MODEL_CONFIG['MODEL_PATH']

    if os.path.exists(vae_path) and os.path.exists(wm_path):
        print("Found pre-trained perception models. Loading them.")
    else:
        print("No pre-trained models found. Performing a quick training session...")
        experiences = collect_experience_data(bootstrap_env, num_steps=1500)
        vae = train_visual_cortex(experiences, epochs=10) 
        world_model = train_world_model(vae, experiences, epochs=15)
        vae.save_weights(vae_path)
        world_model.save_weights(wm_path)
        print("Perception models trained and saved.")
    
    bootstrap_env.close()

    # --- Step 2: Initialize the Embodied Agent & Environment ---
    print("\n--- Initializing Autonomous Embodied Agent ---")
    env = SimpleGridworld(render_mode='human', add_door_key=True)
    
    agent_config = {
        'agent_id': "embodied_agent_v1",
        'verbose': 1,
        'action_space': env.action_space,
        'vae_config': DEFAULT_VAE_CONFIG,
        'world_model_config': DEFAULT_WORLD_MODEL_CONFIG,
        'lifelong_learning_config': DEFAULT_LIFELONG_LEARNING_CONFIG,
    }
    agent = SimplifiedOrchOREmulator(**agent_config)

    # <-- NEW SECTION: LOAD STATE & SET GOALS -->
    # Attempt to load the agent's previous "life"
    agent.load_state() 
    
    # If the agent doesn't have a main quest (e.g., first run or completed last time), give it one.
    if not any(g.goal_type == 'MAIN_QUEST' for g in agent.goal_manager.goals):
        print("Agent has no main quest. Assigning a new one.")
        main_quest = GoalState(
            current_goal="Reach the final target",
            steps=[], 
            base_priority=0.7,
            evaluation_criteria="GOAL_COMPLETION"
        )
        agent.set_goal_state(main_quest)

    # --- Step 3: Run the Main Simulation Loop (with robust save on exit) ---
    try:
        print("\n--- Starting Main Simulation Loop ---")
        obs, info = env.reset()
        reward = 0.0
        terminated = False
        img_obs = env.render()
    
        for cycle in range(500):
            action = agent.run_cycle(img_obs, reward, info, terminated, obs)

            if action is None:
                print(f"[Cycle {agent.current_cycle_num}] Agent chose to think. World is paused.")
                time.sleep(0.5) 
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            img_obs = env.render()

            if terminated or truncated:
                print(f"\n--- Episode Finished (Reason: {'Goal Reached' if reward > 0 else 'Death/Truncated'}) ---")
                # Run one final cycle to process the terminal state
                agent.run_cycle(img_obs, reward, info, terminated, obs)
                break
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n--- Simulation interrupted by user. ---")
    finally:
        # --- Step 4: Final Wrap-up ---
        print("\n--- Simulation Complete ---")
        agent.print_internal_state_summary()
        agent.save_state() # <-- NEW: Save the agent's final state
        env.close()
