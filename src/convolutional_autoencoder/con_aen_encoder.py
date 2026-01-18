from tensorflow.keras import layers, models
from sys import path
from os import path as os_path

# Add parent directory to path to import data_pipeline
path.append(os_path.dirname(os_path.dirname(os_path.abspath(__file__))))
from data_pipeline import IMG_SIZE

# Dimension of the latent space (feature vector)
LATENT_DIM = 512 

# ---------- ENCODER ARCHITECTURE ----------
# The encoder compresses the input image into a lower-dimensional latent representation.
# It uses a series of Convolutional layers with stride 2 to downsample the image.
encoder = models.Sequential([
    # Input Layer
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Block 1: 32 filters
    layers.Conv2D(32, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),        
    
    # Block 2: 64 filters
    layers.Conv2D(64, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Block 3: 128 filters
    layers.Conv2D(128, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Block 4: 256 filters
    layers.Conv2D(256, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Flatten the 3D output to 1D vector
    layers.Flatten(),
    
    # Dense layer to produce the final latent vector
    layers.Dense(LATENT_DIM)
], name="encoder")
