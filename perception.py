# perception.py

"""
This file defines the agent's new deep learning-based sensory and predictive systems.
It contains the Visual Cortex (a Variational Autoencoder) for perception and the
Predictive World Model (an LSTM) for imagination.
"""

# This is the canonical, most robust way to import Keras
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np

# ---------------------------------------------------------------------------
# Visual Cortex - VAE
# ---------------------------------------------------------------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VisualCortexVAE(Model):
    """
    The agent's "eyes." A Variational Autoencoder that learns to compress
    raw pixel observations into a meaningful low-dimensional latent space.
    """
    def __init__(self, original_dim, latent_dim=32, **kwargs):
        super(VisualCortexVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.original_dim = original_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def _build_encoder(self):
        encoder_inputs = layers.Input(shape=self.original_dim)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        # This intermediate shape depends on the output of the encoder's flatten layer
        # For a 64x64 input, after two strides=2 convolutions, the shape is 16x16x64
        x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((16, 16, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        return Model(latent_inputs, decoder_outputs, name="decoder")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, _, _ = self.encoder(inputs)
        return self.decoder(z_mean)

    @tf.function
    def observe_to_latent_vector(self, image: tf.Tensor) -> np.ndarray:
        """The primary inference function. Sees an image, returns a concept vector."""
        # Add a batch dimension if it's a single image
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        z_mean, _, _ = self.encoder(image)
        return z_mean

# ---------------------------------------------------------------------------
# Predictive World Model - LSTM
# ---------------------------------------------------------------------------

class PredictiveWorldModel(Model):
    """
    The agent's "imagination." An LSTM-based model that learns the "physics"
    of the latent space. Given a current latent state and an action, it
    predicts the next latent state.
    """
    def __init__(self, latent_dim, num_actions, **kwargs):
        super(PredictiveWorldModel, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # The model takes two inputs: the latent vector and the one-hot encoded action
        latent_input = layers.Input(shape=(self.latent_dim,), name="latent_input")
        action_input = layers.Input(shape=(self.num_actions,), name="action_input")
        
        # Concatenate inputs and prepare for LSTM
        merged = layers.Concatenate()([latent_input, action_input])
        x = layers.Reshape((1, self.latent_dim + self.num_actions))(merged)
        
        # LSTM layer to model sequence
        x = layers.LSTM(128, return_sequences=False, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        output_latent = layers.Dense(self.latent_dim)(x)
        
        self.model = Model(inputs=[latent_input, action_input], outputs=output_latent, name="world_model")

    def call(self, inputs):
        return self.model(inputs)

    @tf.function
    def predict_next_latent_state(self, current_latent: tf.Tensor, action: int) -> np.ndarray:
        """The primary inference function. Imagines the future."""
        # One-hot encode the action
        action_one_hot = tf.one_hot([action], self.num_actions)

        # Add batch dimension if single sample
        if len(current_latent.shape) == 1:
            current_latent = tf.expand_dims(current_latent, 0)

        prediction = self.model([current_latent, action_one_hot])
        return prediction
