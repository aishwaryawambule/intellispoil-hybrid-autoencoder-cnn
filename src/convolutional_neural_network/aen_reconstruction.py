import tensorflow as tf
from os import path as os_path
import os
import sys
import numpy as np

# ---------- PATH SETUP ----------
SRC_DIR = os_path.dirname(os_path.abspath(__file__))
sys.path.append(os_path.dirname(SRC_DIR))  # Add parent directory to import data_pipeline

from data_pipeline import train_ds, test_ds

MODEL_DIR = os_path.abspath(os_path.join(SRC_DIR, "..", "model"))

# ---------- DATA EXTRACTION ----------
def get_data_and_labels(dataset):
    """
    Efficiently extract images and labels from a tf.data.Dataset.
    Uses .as_numpy_iterator() to avoid building large intermediate lists.
    
    Args:
        dataset (tf.data.Dataset): The input dataset.
        
    Returns:
        tuple: (X, y) where X is the array of images and y is the array of labels.
    """
    images_list = []
    labels_list = []

    for batch in dataset.as_numpy_iterator():
        img, label = batch
        images_list.append(img)
        labels_list.append(label)

    X = np.concatenate(images_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

print("Extracting data and labels from datasets...")
# Load training and testing data into memory
# This is used by cnn_classifier.py to get the data for training
X_train, y_train = get_data_and_labels(train_ds)
X_test, y_test = get_data_and_labels(test_ds)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------- RECONSTRUCTION PLACEHOLDER ----------
# The following section is commented out but shows how one might generate reconstructions
# using the autoencoder if needed.
print("Generating reconstructed images...")
# X_train_recon = autoencoder_model.predict(X_train)
# X_test_recon  = autoencoder_model.predict(X_test)
print("Reconstruction complete.")
