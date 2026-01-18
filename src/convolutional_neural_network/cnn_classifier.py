import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- SYSTEM SAFETY ----------
# Allow duplicate OpenMP libraries to prevent conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Enable GPU memory growth to avoid allocating all memory at once
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ---------- DATA IMPORT ----------
# Import pre-processed data from aen_reconstruction script
# This assumes that aen_reconstruction.py performs the necessary data loading and preprocessing
try:
    from aen_reconstruction import X_train, X_test, y_train, y_test
    print("Data loaded successfully")
    print("Train shape:", X_train.shape)
except ImportError as e:
    print("Data import failed:", e)
    sys.exit(1)

# ---------- LOAD ENCODER ----------
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
ENCODER_PATH = os.path.join(MODEL_DIR, "con_aen_encoder_model.keras")

if not os.path.exists(ENCODER_PATH):
    print(f"Error: Encoder model not found at {ENCODER_PATH}")
    print("Please train the autoencoder first using 'src/convolutional_autoencoder/con_aen.py'")
    sys.exit(1)

print(f"Loading encoder from: {ENCODER_PATH}")
encoder = tf.keras.models.load_model(ENCODER_PATH)
encoder.trainable = False  # Freeze encoder to use it as a feature extractor

# ---------- MODEL DEFINITION ----------
# Define the input shape based on the training data
INPUT_SHAPE = X_train.shape[1:]  # safer than hardcoding

inputs = layers.Input(shape=INPUT_SHAPE)

# Encoder: Extract features from the input image
x = encoder(inputs)

# Ensure proper shape for the dense layers
# If the encoder output is 4D (batch, height, width, channels), pool it to 2D
if len(x.shape) == 4:
    x = layers.GlobalAveragePooling2D()(x)

# Classification Head: Fully connected layers for binary classification
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# Output layer: Sigmoid activation for binary classification (0 or 1)
outputs = layers.Dense(1, activation="sigmoid")(x)

cnn = models.Model(inputs, outputs)

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

cnn.summary()

# ---------- TRAINING ----------
# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("Starting training...")
history = cnn.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# ---------- SAVE MODEL ----------
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, "cnn_classifier_model.keras")
cnn.save(SAVE_PATH)
print(f"Model saved successfully at {SAVE_PATH}")
