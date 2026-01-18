import os
from sys import path
from os import path as os_path
import tensorflow as tf
from tensorflow.keras import layers, models

# Import encoder and decoder models
# Note: These imports assume that con_aen_encoder.py and con_aen_decoder.py are in the same directory
from con_aen_encoder import encoder
from con_aen_decoder import decoder

# Import data pipeline
# Note: data_pipeline.py is in the parent directory, so we need to ensure it's accessible.
# The original code relied on implicit path setup or running from a specific location.
# It's better to be explicit if possible, but for now we'll stick to the existing import structure
# which seems to rely on the script being run in a way that data_pipeline is importable.
from data_pipeline import train_ds_aen, test_ds_aen, IMG_SIZE

# ---------- MODEL ASSEMBLY ----------
# Input layer for the autoencoder
autoencoder_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Connect Encoder and Decoder
latent = encoder(autoencoder_input)
# encoder.summary()

restructured = decoder(latent)
# decoder.summary()

# Create the full Autoencoder model
autoencoder_model = models.Model(autoencoder_input, restructured)

# Compile the model
# Using Mean Squared Error (MSE) as the loss function for reconstruction
autoencoder_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse'
)

autoencoder_model.summary()

# ---------- TRAINING ----------
EPOCHS = 100

print("Starting Autoencoder training...")
autoencoder_model.fit(
    train_ds_aen,
    epochs=EPOCHS,
    validation_data=test_ds_aen
)

# ---------- SAVE MODEL ----------
# Define model directory relative to this script
SRC_DIR = os_path.dirname(os_path.abspath(__file__))
MODEL_DIR = os_path.join(SRC_DIR, "..", "model")

# Ensure the model directory exists
if not os_path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"Saving models to {MODEL_DIR}...")

# Save full autoencoder
autoencoder_model.save_weights(os_path.join(MODEL_DIR, "con_aen.weights.h5"))
autoencoder_model.save(os_path.join(MODEL_DIR, "con_aen_model.keras"))

# Save encoder separately (for use in classifier)
encoder.save_weights(os_path.join(MODEL_DIR, "con_aen_encoder.weights.h5"))
encoder.save(os_path.join(MODEL_DIR, "con_aen_encoder_model.keras"))

# Save decoder separately
decoder.save_weights(os_path.join(MODEL_DIR, "con_aen_decoder.weights.h5"))
decoder.save(os_path.join(MODEL_DIR, "con_aen_decoder_model.keras"))

print("Models saved successfully.")

# Freeze encoder after training (optional, but good practice if used elsewhere)
encoder.trainable = False
