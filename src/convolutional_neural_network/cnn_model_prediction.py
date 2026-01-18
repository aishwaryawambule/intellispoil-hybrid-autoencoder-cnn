import tensorflow as tf
from os import path as os_path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set figure size for plots
plt.figure(figsize=(15, 5))

# Add parent directory to path to import data_pipeline
sys.path.append(os_path.dirname(os_path.dirname(os_path.abspath(__file__))))
from data_pipeline import load_dataset, normalize

DATASET_PATH = os_path.join(os_path.dirname(os_path.dirname(os_path.abspath(__file__))), "data/prediction_use_dataset")

SRC_DIR = os_path.dirname(os_path.abspath(__file__))
MODEL_PATH = os_path.join(SRC_DIR, "../model/cnn_classifier_model.keras")

# 1. Load Model
print("Loading model...")
if not os_path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please train the classifier first using 'src/convolutional_neural_network/cnn_classifier.py'")
    sys.exit(1)

# The classifier now contains the encoder internally, so we only need to load the classifier
cnn_classifier_model = tf.keras.models.load_model(MODEL_PATH)

# 2. Load Banana Data
print("\nLoading Banana Dataset for Prediction...")
# We specifically load 'banana' images to test generalization or specific performance
banana_test_ds = load_dataset(os_path.join(DATASET_PATH, "Test"), "banana")
banana_test_ds = banana_test_ds.map(normalize)

# 3. Extract Banana Images
def get_data_and_labels(dataset):
    """
    Extracts all images and labels from a dataset into numpy arrays.
    """
    images = []
    labels = []
    # Use as_numpy_iterator for cleaner extraction
    for img, label in dataset.as_numpy_iterator():
        images.append(img)
        labels.append(label)
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

X_banana, y_banana = get_data_and_labels(banana_test_ds)
print(f"Extracted {len(X_banana)} banana samples.")

if len(X_banana) == 0:
    print("No banana samples found. Exiting.")
    sys.exit(0)

# 4. Perform Predictions
print("\nSample Predictions on Banana Data:")
num_samples = 5
# Ensure we don't try to sample more than we have
num_samples = min(num_samples, len(X_banana))
indices = np.random.choice(len(X_banana), num_samples, replace=False)

for idx, i in enumerate(indices):
    img = X_banana[i:i+1]
    true_label = "Rotten" if y_banana[i] == 1 else "Healthy"
    
    # Pass original image directly to classifier (it handles encoding internally)
    prediction = cnn_classifier_model.predict(img, verbose=0)[0][0]
    pred_label = "Rotten" if prediction > 0.5 else "Healthy"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"Sample {i}: True={true_label}, Pred={pred_label} ({confidence*100:.2f}% confidence)")

    plt.subplot(1, num_samples, idx+1)
    # img is (1, 450, 450, 3) and float [0,1]. Squeeze batch dim and show.
    plt.imshow(np.squeeze(img)) 
    plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence*100:.1f}%", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()


# 5. Overall Evaluation on Banana Data
# print("\nOverall Evaluation on Banana Test Set:")
# loss, acc = cnn_classifier_model.evaluate(X_banana, y_banana, verbose=1)
# print(f"Banana Accuracy: {acc*100:.2f}%")
