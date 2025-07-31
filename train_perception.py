# train_perception.py

"""
This script contains the functions to bootstrap the agent's perception models.
It handles collecting raw visual data from the environment and training both the
VisualCortexVAE and the PredictiveWorldModel.
"""
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm

from environment import SimpleGridworld
from perception import VisualCortexVAE, PredictiveWorldModel

# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------
def collect_experience_data(env: SimpleGridworld, num_steps=10000, img_size=(64, 64)):
    """Runs the environment with random actions to collect a dataset of experiences."""
    print(f"Collecting {num_steps} steps of experience data...")
    experiences = []
    
    # Resize the rendering window to our target image size for efficiency
    env.window_size = img_size[0]
    
    obs, _ = env.reset()
    # Initial render to ensure window is created if in 'human' mode, then switch to 'rgb_array'
    if env.render_mode == "human":
        env.render()
    env.render_mode = 'rgb_array'

    # Manually resize the observation from reset before the loop starts
    frame = env.render()
    current_state_img = tf.image.resize(frame, img_size) / 255.0
    
    for _ in tqdm(range(num_steps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        next_state_img = tf.image.resize(frame, img_size) / 255.0
        
        experiences.append({
            'state': current_state_img.numpy(),
            'action': action,
            'next_state': next_state_img.numpy(),
        })

        current_state_img = next_state_img
        if terminated or truncated:
            obs, _ = env.reset()
            frame = env.render()
            current_state_img = tf.image.resize(frame, img_size) / 255.0

    print("Data collection complete.")
    return experiences

# ---------------------------------------------------------------------------
# VAE Training
# ---------------------------------------------------------------------------
def train_visual_cortex(experiences, img_size=(64, 64), latent_dim=32, epochs=30, batch_size=128):
    """Trains the VisualCortexVAE on the collected image data."""
    print("--- Training Visual Cortex (VAE) ---")
    
    images = np.array([exp['state'] for exp in experiences])
    if images.shape[-1] > 3:
        images = images[...,:3]

    print(f"Dataset shape: {images.shape}")
    
    vae = VisualCortexVAE(original_dim=(*img_size, 3), latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam())    
    # Build the model by passing a single data point before training.
    if not vae.built:
        vae.build(input_shape=(None, *img_size, 3))
    
    vae.fit(images, epochs=epochs, batch_size=batch_size)
    
    print("VAE training complete.")
    return vae

# ---------------------------------------------------------------------------
# World Model Training
# ---------------------------------------------------------------------------
def train_world_model(vae: VisualCortexVAE, experiences, latent_dim=32, num_actions=4, epochs=40, batch_size=128):
    """
    Trains the PredictiveWorldModel using the pre-trained VAE to convert images
    to latent vectors.
    """
    print("--- Training Predictive World Model ---")
    
    print("Converting image data to latent space...")
    states_imgs = np.array([exp['state'] for exp in experiences])
    next_states_imgs = np.array([exp['next_state'] for exp in experiences])
    actions = np.array([exp['action'] for exp in experiences])

    latent_states = vae.observe_to_latent_vector(states_imgs).numpy()
    latent_next_states = vae.observe_to_latent_vector(next_states_imgs).numpy()
    
    print(f"Latent state dataset shape: {latent_states.shape}")

    actions_one_hot = tf.one_hot(actions, num_actions)
    dataset_X = [latent_states, actions_one_hot]
    dataset_y = latent_next_states
    
    world_model = PredictiveWorldModel(latent_dim=latent_dim, num_actions=num_actions)
    world_model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

    # Build the world model before training.
    if not world_model.built:
        world_model.build(input_shape=[(None, latent_dim), (None, num_actions)])
    
    world_model.fit(dataset_X, dataset_y, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    print("World Model training complete.")
    return world_model