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



    for i_episode in range(num_episodes):
        print(f"\n======== Starting Episode {i_episode + 1}/{num_episodes} ========")

        # 1. Reset the world and get the initial, full state observation.
        current_full_obs, info = env.reset()
        current_image_obs = env.render()
        
        # 2. Initialize loop variables.
        reward = 0.0
        terminated = False
        truncated = False
        
        # 3. Set the agent's main, persistent goal for the episode.
        agent.set_goal_state(GoalState(
            current_goal=f"Solve Maze Task (E{i_episode+1})",
            goal_type="MAIN_QUEST",
            base_priority=0.4,
            steps=[], # Let the agent decompose the task itself
            evaluation_criteria="GOAL_COMPLETION"
        ))

        # 4. Run the episode until termination or timeout.
        completion_reason = "Timeout"  # Default reason if the loop finishes
        t = 0
        for t in range(400): # More steps to allow for juggling priorities
            
            # --- AGENT'S TURN: PERCEIVE, THINK, ACT ---
            action_to_take = agent.run_cycle(
                raw_image_observation=current_image_obs,
                last_reward=reward,
                last_info=info,
                is_terminated=(terminated or truncated), # Pass a single 'done' flag
                current_full_obs=current_full_obs
            )

            # --- ENVIRONMENT'S TURN: UPDATE ---
            if action_to_take is not None:
                current_full_obs, reward, terminated, truncated, info = env.step(action_to_take)
                current_image_obs = env.render()
            else: # Agent is thinking
                reward = 0.0
                terminated = False
                truncated = False
                info = {}

            # --- CHECK FOR EPISODE END ---
            main_goal_is_complete = agent.is_main_goal_completed()

            if terminated:
                completion_reason = "Goal Reached!"
                break
            if truncated:
                completion_reason = "Truncated"
                break
            if main_goal_is_complete:
                completion_reason = "Agent Reported Goal Complete"
                break
        
        # --- End of Episode Processing ---
        # Allow the agent to process the final state of the episode.
        agent.run_cycle(
            raw_image_observation=current_image_obs,
            last_reward=reward,
            last_info=info,
            is_terminated=(terminated or truncated or main_goal_is_complete),
            current_full_obs=current_full_obs
        )
        print(f"Episode {i_episode+1} finished after {t+1} timesteps. ({completion_reason})")


        # --- End of Episode Sleep/Training Logic ---
        ll_params = agent.ll_params
        if ll_params.get('enabled', False) and len(agent.experience_replay_buffer) >= ll_params.get('training_batch_size', 64):
            print(f"\n[Cognitive Cycle] End of episode. Consolidating memories through sleep/training...")
            if agent.sleep_train():
                print("[Cognitive Cycle] ...Training complete. Perception models updated.")
                agent.visual_cortex.save_weights(agent.vae_params['MODEL_PATH'])
                agent.world_model.save_weights(agent.world_model_params['MODEL_PATH'])
                agent.experience_replay_buffer.buffer.clear()
        
        time.sleep(1) # Pause between episodes

    env.close()
    print("\\n--- Agent's Life Simulation Completed ---")
