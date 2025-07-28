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
import collections
from typing import Dict, List, Tuple, Any

from core_abstractions import ActiveConceptNetGraph, SymbolicObject


class SymbolicPerceptionCore:
    """
    Transforms raw, chaotic data (grids) into a structured, relational format
    (an ActiveConceptNetGraph) ripe for reasoning. This is the agent's
    laboratory and measurement toolkit for formal reasoning tasks like ARC.
    """

    def _segment_objects(self, grid: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
        """
        Segments the grid into distinct objects. This is a crucial perception step.
        CORRECTED: Now segments based on connected components of the SAME color.
        """
        if grid is None or grid.size == 0:
            return {}

        rows, cols = grid.shape
        visited = set()
        objects = {}
        object_id_counter = 1
        background_color = 0  # Standard ARC background assumption

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r, c] != background_color:
                    obj_coords = []
                    current_color = grid[r, c] # Get the color of the new potential object
                    q = collections.deque([(r, c)])
                    visited.add((r, c))
                    
                    while q:
                        row, col = q.popleft()
                        obj_coords.append((row, col))
                        
                        # Check neighbors
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = row + dr, col + dc
                            
                            # The crucial new condition: check if the neighbor is the SAME color
                            if 0 <= nr < rows and 0 <= nc < cols and \
                               (nr, nc) not in visited and grid[nr, nc] == current_color:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                                
                    objects[object_id_counter] = obj_coords
                    object_id_counter += 1
        return objects

    def _extract_intrinsic_properties(self, grid: np.ndarray, obj_coords: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Extracts basic, intrinsic properties of a single object."""
        if not obj_coords:
            return {}
        
        # Color: For single-color objects, this is simple.
        # For multi-color, could be a list or the most common color.
        # Here we assume the color of the first pixel is the object's color.
        first_pixel_color = int(grid[obj_coords[0][0], obj_coords[0][1]])

        # Bounding Box
        min_r = min(r for r, c in obj_coords)
        max_r = max(r for r, c in obj_coords)
        min_c = min(c for r, c in obj_coords)
        max_c = max(c for r, c in obj_coords)

        return {
            'Color': first_pixel_color,
            'Size': len(obj_coords),
            'Position': (min_r, min_c),
            'BoundingBox': {'y':min_r, 'x': min_c, 'height': max_r - min_r + 1, 'width': max_c - min_c + 1}
        }

    def _extract_geometric_topological_properties(self, grid: np.ndarray, obj_coords: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Extracts more complex shape and topological properties."""
        props = {}
        if not obj_coords: return props
        
        min_r, min_c = min(r for r,c in obj_coords), min(c for r,c in obj_coords)
        
        # Shape Hash (canonical representation)
        relative_coords = sorted([(r - min_r, c - min_c) for r, c in obj_coords])
        props['ShapeHash'] = tuple(relative_coords)
        
        # Symmetry (Simplified for this example)
        # True horizontal symmetry check is more complex. This is a basic check.
        height = max(r for r,c in obj_coords) - min_r + 1
        width = max(c for r,c in obj_coords) - min_c + 1
        obj_grid = np.zeros((height, width))
        for r,c in relative_coords: obj_grid[r,c] = 1
        
        props['Symmetry_Horizontal'] = np.array_equal(obj_grid, np.flip(obj_grid, axis=0))
        props['Symmetry_Vertical'] = np.array_equal(obj_grid, np.flip(obj_grid, axis=1))
        # Note: Rotational symmetry is non-trivial and omitted for this iteration.

        # Topology (Simplified)
        props['IsContiguous'] = True # By definition of our segmentation
        # IsHollow check would require another flood-fill from the background within the bounding box
        props['IsHollow'] = False # Placeholder

        return props

    def _extract_global_properties(self, grid: np.ndarray, object_count: int) -> Dict[str, Any]:
        """Derives properties of the entire scene."""
        if grid.size == 0:
            return {}

        # A more robust way to determine background color is the most frequent color.
        # This is a common and effective heuristic for ARC tasks.
        color_counts = collections.Counter(grid.flatten())
        background_color = color_counts.most_common(1)[0][0]

        # The color palette should only include colors of FOREGROUND objects.
        # So, we take all unique colors and remove the background color.
        unique_colors = np.unique(grid)
        foreground_palette = [int(c) for c in unique_colors if c != background_color and c != 0]

        # --- THE CRUCIAL HEURISTIC ---
        # If the grid is monochromatic (only one color besides black/0) AND
        # the perception system only found one "object", it's a solid fill.
        # In this case, the foreground palette should be empty.
        if len(unique_colors) <= 2 and 0 in unique_colors and object_count == 1:
            # This handles cases where the grid is all one color on a black background.
            # We check if that one color is the same as the background color we just determined.
            single_color_present = next((c for c in unique_colors if c != 0), None)
            if single_color_present == background_color:
                foreground_palette = []

        # Also handle the case where the grid is *only* one color (no 0s)
        if len(unique_colors) == 1 and unique_colors[0] != 0:
            foreground_palette = []

        return {
            'GridSymmetry_H': np.array_equal(grid, np.flip(grid, axis=0)),
            'GridSymmetry_V': np.array_equal(grid, np.flip(grid, axis=1)),
            'BackgroundColor': int(background_color),
            'ObjectCount': object_count if foreground_palette else 0, # Report 0 objects if it's a fill
            'ColorPalette': foreground_palette
        }
    
    def _compute_relations(self, graph: ActiveConceptNetGraph):
        """Computes and adds relational edges between object nodes."""
        obj_nodes = [node for node in graph.nodes.values() if node.type == 'object']
        for i in range(len(obj_nodes)):
            for j in range(i + 1, len(obj_nodes)):
                obj1 = obj_nodes[i]
                obj2 = obj_nodes[j]

                pos1 = obj1.attributes.get('Position', (0,0))
                pos2 = obj2.attributes.get('Position', (0,0))
                
                # RelativePosition
                graph.add_edge(obj1.id, obj2.id, 'RelativePosition', dx=pos2[1]-pos1[1], dy=pos2[0]-pos1[0])

                # Alignment
                if pos1[0] == pos2[0]:
                    graph.add_edge(obj1.id, obj2.id, 'Alignment', type='Horizontal')
                if pos1[1] == pos2[1]:
                    graph.add_edge(obj1.id, obj2.id, 'Alignment', type='Vertical')

    def parse(self, grid: np.ndarray) -> ActiveConceptNetGraph:
        """
        The main function of this core. Runs the full parsing gauntlet to
        convert a raw grid into a rich ActiveConceptNetGraph.
        """
        graph = ActiveConceptNetGraph()
        segmented_objects = self._segment_objects(grid)

        # Layer 3: Global Properties
        global_props = self._extract_global_properties(grid, len(segmented_objects))
        grid_node = SymbolicObject(id='grid', type='grid', attributes=global_props)
        graph.add_node(grid_node)

        # Layers 1 & 2: Object Segmentation and Property Extraction
        for obj_id, obj_coords in segmented_objects.items():
            node_id = f"obj_{obj_id}"
            intrinsic_props = self._extract_intrinsic_properties(grid, obj_coords)
            geometric_props = self._extract_geometric_topological_properties(grid, obj_coords)
            all_props = {**intrinsic_props, **geometric_props, 'Coordinates': obj_coords}

            obj_node = SymbolicObject(id=node_id, type='object', attributes=all_props)
            graph.add_node(obj_node)
        
        # Post-processing: Compute relations between all extracted objects
        self._compute_relations(graph)
        
        return graph
    
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
    @tf.function
    def call(self, inputs):
        return self.model(inputs)

    def predict_next_latent_state(self, current_latent: tf.Tensor, action: int) -> np.ndarray:
        """The primary inference function. Imagines the future."""
        # One-hot encode the action
        action_one_hot = tf.one_hot([action], self.num_actions)

        # Add batch dimension if single sample
        if len(current_latent.shape) == 1:
            current_latent = tf.expand_dims(current_latent, 0)

        prediction = self.model([current_latent, action_one_hot])
        return prediction