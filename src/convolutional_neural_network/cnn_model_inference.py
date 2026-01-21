import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PATH SETUP
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data_pipeline import load_dataset, normalize

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "model"
CNN_MODEL_PATH = MODEL_DIR / "cnn_classifier_model.keras"
DATASET_PATH = PROJECT_ROOT / "data" / "prediction_use_dataset"

# Folder to save results
RESULTS_DIR = PROJECT_ROOT / "../results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_PATH = RESULTS_DIR / "_inference_samples.png"

# =========================================================
# LOAD MODEL
# =========================================================
print("Loading CNN classifier model...")
if not CNN_MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {CNN_MODEL_PATH}")

cnn_classifier_model = tf.keras.models.load_model(CNN_MODEL_PATH)

# =========================================================
# DYNAMIC FRUIT/VEGETABLE NAME (default: banana)
# =========================================================
import argparse

parser = argparse.ArgumentParser(description="Run CNN inference on a fruit/vegetable dataset")
parser.add_argument("--name", type=str, default="banana", help="Name of the fruit/vegetable folder")
args = parser.parse_args()

name = args.name.lower()
print(f"\nRunning inference on '{name}' samples...")

# =========================================================
# LOAD TEST DATA
# =========================================================
test_ds = load_dataset(DATASET_PATH / "Test", name)
test_ds = test_ds.map(normalize)

# =========================================================
# EXTRACT DATA
# =========================================================
def get_data_and_labels(dataset):
    images, labels = [], []
    for img, label in dataset:
        images.append(img)
        labels.append(label)
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

X_test, y_test = get_data_and_labels(test_ds)

if len(X_test) == 0:
    raise RuntimeError(f"No samples found for '{name}'.")

print(f"Extracted {len(X_test)} {name} samples.")

# =========================================================
# SAMPLE PREDICTIONS & VISUALIZATION
# =========================================================
NUM_SAMPLES = min(5, len(X_test))
THRESHOLD = 0.7

indices = np.random.choice(len(X_test), NUM_SAMPLES, replace=False)

plt.figure(figsize=(15, 4))

for idx, i in enumerate(indices):
    img = X_test[i:i + 1]
    true_label = "Rotten" if y_test[i] == 1 else "Healthy"

    prob = cnn_classifier_model.predict(img, verbose=0)[0][0]
    pred_label = "Rotten" if prob > THRESHOLD else "Healthy"
    confidence = prob if prob > THRESHOLD else 1.0 - prob

    print(
        f"Sample {i}: True={true_label}, "
        f"Pred={pred_label}, "
        f"Confidence={confidence * 100:.2f}%"
    )

    plt.subplot(1, NUM_SAMPLES, idx + 1)
    plt.imshow(np.squeeze(img))
    plt.title(
        f"True: {true_label}\n"
        f"Pred: {pred_label}\n"
        f"Conf: {confidence * 100:.1f}%",
        fontsize=9
    )
    plt.axis("off")

plt.tight_layout()

# =========================================================
# SAVE & SHOW PLOT
# =========================================================
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"\nPlot saved to: {PLOT_PATH}")
