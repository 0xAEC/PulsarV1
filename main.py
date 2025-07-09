# main.py
import os
import numpy as np
import random
import time
import copy
import pygame

from environment import SimpleGridworld
from cognitive_engine import SimplifiedOrchOREmulator
from core_abstractions import GoalState
from configurations import DEFAULT_VAE_CONFIG, DEFAULT_WORLD_MODEL_CONFIG, DEFAULT_LIFELONG_LEARNING_CONFIG

# Import training functions
from train_perception import collect_experience_data, train_visual_cortex, train_world_model

def bootstrap_perception_models():
    """
    Checks for pre-trained models. If they don't exist, it orchestrates the
    data collection and training process from scratch.
    """
    # Use the corrected filenames
    vae_path = 'visual_cortex.weights.h5'
    world_model_path = 'world_model.weights.h5'

    # Check if the files already exist
    if os.path.exists(vae_path) and os.path.exists(world_model_path):
        print("--- Found pre-trained perception model weights. Skipping bootstrap. ---")
        # Update the global config to use these paths for loading
        DEFAULT_VAE_CONFIG['MODEL_PATH'] = vae_path
        DEFAULT_WORLD_MODEL_CONFIG['MODEL_PATH'] = world_model_path
        return

    print("--- Perception models not found. Starting bootstrap process... ---")
    bootstrap_env = SimpleGridworld(size=15, n_obstacles=12, render_mode='rgb_array')

    experiences = collect_experience_data(bootstrap_env, num_steps=5000)
    bootstrap_env.close()

    vae = train_visual_cortex(
        experiences,
        img_size=DEFAULT_VAE_CONFIG['IMG_SIZE'],
        latent_dim=DEFAULT_VAE_CONFIG['LATENT_DIM'],
        epochs=30
    )
    vae.save_weights(vae_path)
    print(f"VAE model saved to {vae_path}")
    
    # Update the config to use the new paths for the main agent
    DEFAULT_VAE_CONFIG['MODEL_PATH'] = vae_path
    DEFAULT_WORLD_MODEL_CONFIG['MODEL_PATH'] = world_model_path

    world_model = train_world_model(
        vae,
        experiences,
        latent_dim=DEFAULT_VAE_CONFIG['LATENT_DIM'],
        num_actions=bootstrap_env.action_space.n,
        epochs=40
    )
    world_model.save_weights(world_model_path)
    print(f"World Model saved to {world_model_path}")
    print("--- Bootstrap complete. ---")

# ---------------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if 'XDG_RUNTIME_DIR' not in os.environ:
        dummy_xdg_dir = '/tmp/xdg_runtime_dir_fallback'
        os.makedirs(dummy_xdg_dir, exist_ok=True)
        os.chmod(dummy_xdg_dir, 0o700)
        os.environ['XDG_RUNTIME_DIR'] = dummy_xdg_dir

    seed_val = int(time.time())
    np.random.seed(seed_val)
    random.seed(seed_val)
    print(f"Global random seed set to: {seed_val}")

    bootstrap_perception_models()

    print("\n--- Initializing Environment for Agent's Life ---")
    GRID_SIZE = 15
    N_OBSTACLES = 12
    env = SimpleGridworld(size=GRID_SIZE, n_obstacles=N_OBSTACLES, render_mode="human")

    print("\n--- Initializing Autonomous Cognitive Agent ---")
    agent_config = {
        'agent_id': "autonomous_agent_v1",
        'verbose': 1,
        'action_space': env.action_space,
        'vae_config': DEFAULT_VAE_CONFIG,
        'world_model_config': DEFAULT_WORLD_MODEL_CONFIG,
        'lifelong_learning_config': DEFAULT_LIFELONG_LEARNING_CONFIG,
    }
    agent = SimplifiedOrchOREmulator(**agent_config)
    
    print("\n--- Starting Main Life-Cycle Loop ---")
    num_episodes = 10
    last_action = None

    # === THE CANONICAL, CORRECT MAIN LOOP ===

    for i_episode in range(num_episodes):
        print(f"\\n======== Starting Episode {i_episode + 1}/{num_episodes} ========")

        # 1. Reset the world and get the very first observation.
        obs, info = env.reset()
        current_image_obs = env.render()
        
        # Initialize loop variables
        reward = 0.0
        terminated = False
        
        agent.set_goal_state(GoalState(
            current_goal=f"Solve the maze (Episode {i_episode + 1})",
            steps=[],
            evaluation_criteria="GOAL_COMPLETION"
        ))
        print(f"Agent assigned high-level goal: '{agent.current_goal_state_obj.current_goal}'")


        for t in range(250): # Give it a bit more time per episode
            # --- 1. Agent Perceives and Acts ---
            # The agent's cognitive cycle takes in the result of the LAST action
            # and the CURRENT view of the world, then decides on the NEXT action.
            action_to_take = agent.run_cycle(current_image_obs, reward, info, terminated)

            # --- 2. The World Responds ---
            # If the agent decides on a physical action, we apply it to the environment.
            if action_to_take is not None:
                obs, reward, terminated, truncated, info = env.step(action_to_take)
                current_image_obs = env.render() # Update the view for the next loop
            else:
                # Agent chose to "think" or couldn't decide. The world doesn't change.
                # Reward is 0, nothing has terminated, info is empty.
                reward = 0.0
                terminated = False
                info = {}
                # The visual observation remains the same as the last step.

            # Optional: Add the visualization call here if you want it
            # draw_minds_eye(agent, mind_window, mind_font)
            
            # --- 3. Check for Episode End ---
            if terminated:
                break
        
        # --- End of Episode Processing ---
        if terminated:
            # Process the final successful state one last time
            agent.run_cycle(current_image_obs, reward, info, terminated)
            print(f"Episode {i_episode+1} finished after {t+1} timesteps. Goal Reached!")
        else:
            print(f"Episode {i_episode+1} finished after {t+1} timesteps. (Timeout)")

        # --- End of Episode Sleep/Training Logic ---
        ll_params = agent.ll_params
        if ll_params.get('enabled', False) and len(agent.experience_replay_buffer) >= ll_params.get('training_batch_size', 64):
            print(f"\\n[Cognitive Cycle] End of episode. Consolidating memories through sleep/training...")
            if agent.sleep_train():
                print("[Cognitive Cycle] ...Training complete. Perception models updated.")
                agent.visual_cortex.save_weights(agent.vae_params['MODEL_PATH'])
                agent.world_model.save_weights(agent.world_model_params['MODEL_PATH'])
                agent.experience_replay_buffer.buffer.clear()
        
        time.sleep(1) # Pause between episodes

    env.close()
    print("\\n--- Agent's Life Simulation Completed ---")
