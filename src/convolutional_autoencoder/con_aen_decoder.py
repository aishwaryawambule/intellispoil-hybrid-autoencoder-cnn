from tensorflow.keras import layers, models
from sys import path
from os import path as os_path
import math

# Add parent directory to path to import data_pipeline
path.append(os_path.dirname(os_path.dirname(os_path.abspath(__file__))))
from data_pipeline import IMG_SIZE

LATENT_DIM = 512 

# Calculate the bottleneck dimension after 4 layers of stride 2
# This is needed to reshape the dense output back into a 3D volume
bottleneck_dim = IMG_SIZE
for _ in range(4):
    bottleneck_dim = math.ceil(bottleneck_dim / 2)

# ---------- DECODER ARCHITECTURE ----------
# The decoder reconstructs the image from the latent representation.
# It mirrors the encoder structure using Conv2DTranspose layers.
decoder = models.Sequential([
    layers.Input(shape=(LATENT_DIM,)),

    # Expand latent vector to a large enough volume
    layers.Dense(bottleneck_dim * bottleneck_dim * 256),
    layers.Reshape((bottleneck_dim, bottleneck_dim, 256)),

    # Block 1: Upsample
    layers.Conv2DTranspose(128, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Block 2: Upsample
    layers.Conv2DTranspose(64, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Block 3: Upsample
    layers.Conv2DTranspose(32, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Output Layer: Reconstruct to original image size and channels
    # Sigmoid activation ensures output values are in [0, 1]
    layers.Conv2DTranspose(3, 3, strides=2, padding="same",
                           activation="sigmoid"),
    
    # Ensure the output size matches IMG_SIZE exactly (in case of rounding issues)
    layers.Resizing(IMG_SIZE, IMG_SIZE)
], name="decoder")
