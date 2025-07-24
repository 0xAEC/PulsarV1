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
    # In a real scenario, you might do this once and save the weights.
    # For this demo, we'll do a quick pre-training session.
    print("\n--- Bootstrapping Agent's Perception & World Model ---")
    bootstrap_env = SimpleGridworld(render_mode='rgb_array')
    
    # Check if pre-trained models exist
    vae_path = DEFAULT_VAE_CONFIG['MODEL_PATH']
    wm_path = DEFAULT_WORLD_MODEL_CONFIG['MODEL_PATH']

    if os.path.exists(vae_path) and os.path.exists(wm_path):
        print("Found pre-trained perception models. Loading them.")
        # We don't need to load them here; the agent's __init__ will handle it.
        pass
    else:
        print("No pre-trained models found. Performing a quick training session...")
        # Collect a small dataset for demonstration purposes
        experiences = collect_experience_data(bootstrap_env, num_steps=1500)
        # Train the "eyes"
        vae = train_visual_cortex(experiences, epochs=10) 
        # Train the "imagination"
        world_model = train_world_model(vae, experiences, epochs=15)
        # Save the models for future runs
        vae.save_weights(vae_path)
        world_model.save_weights(wm_path)
        print("Perception models trained and saved.")
    
    bootstrap_env.close()

    # --- Step 2: Initialize the Embodied Agent ---
    print("\n--- Initializing Autonomous Embodied Agent ---")
    
    # The agent now needs a body (action_space) and its perception systems
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

    # --- Step 3: Give the Agent a High-Level Goal ---
    # The agent must figure out the steps itself (acquire key, open door, reach target).
    main_quest = GoalState(
        current_goal="Reach the final target",
        steps=[], # The agent must discover the steps via its reasoning!
        base_priority=0.7,
        evaluation_criteria="GOAL_COMPLETION"
    )
    agent.set_goal_state(main_quest)

    # --- Step 4: Run the Main Simulation Loop ---
    print("\n--- Starting Main Simulation Loop ---")
    obs, info = env.reset()
    reward = 0.0
    terminated = False
    
    # We need to get the first image observation for the agent's first cycle
    img_obs = env.render()
    
    # The agent runs for a maximum number of steps
    for cycle in range(500):
        # The agent's core cognitive-motor loop
        action = agent.run_cycle(img_obs, reward, info, terminated, obs)

        # If the agent decides to "think" instead of act, we skip the env step
        if action is None:
            print(f"[Cycle {cycle+1}] Agent chose to think. World is paused.")
            time.sleep(0.5) # Pause to make "thinking" visible
            # In the next loop, the agent will see the same image but with an updated internal state
            continue

        # If the agent chose an action, apply it to the world
        obs, reward, terminated, truncated, info = env.step(action)
        img_obs = env.render() # Get the visual result of the action

        if terminated or truncated:
            print(f"\n--- Episode Finished (Reason: {'Goal Reached' if reward > 0 else 'Death/Truncated'}) ---")
            break
            
        # Give a moment to observe the agent's action
        time.sleep(0.1)

    print("\n--- Simulation Complete ---")
    agent.print_internal_state_summary()
    env.close()
