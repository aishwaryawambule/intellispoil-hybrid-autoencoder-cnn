import os
import sys
from os import path as os_path
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

SRC_DIR = os_path.dirname(os_path.abspath(__file__))
sys.path.append(os_path.dirname(SRC_DIR))  # Add src directory to path

# Import data
try:
    from aen_reconstruction import X_test, y_test
    print("Data loaded successfully")
except ImportError:
    try:
        from convolutional_neural_network.aen_reconstruction import X_test, y_test
        print("Data loaded successfully")
    except ImportError as e:
        print(f"Error importing data: {e}")
        sys.exit(1)

MODEL_DIR = os_path.abspath(os_path.join(SRC_DIR, "..", "model"))
MODEL_PATH = os_path.join(MODEL_DIR, "cnn_classifier_model.keras")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please train the classifier first using 'src/convolutional_neural_network/cnn_classifier.py'")
    sys.exit(1)

print(f"Loading model from {MODEL_PATH}...")
cnn = tf.keras.models.load_model(MODEL_PATH)

# ---------- EVALUATION ----------

# Create results directory
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Evaluating model...")

# Predict on test set
y_pred_prob = cnn.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=["Healthy", "Rotten"])
print("Classification Report:")
print(report)

# Save Classification Report to file
with open(os.path.join(RESULTS_DIR, "cnn_classification_report.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Rotten"])
disp.plot(cmap=plt.cm.Blues)
plt.title("CNN Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png"))
plt.close()

# Load history
HISTORY_PATH = os.path.join(MODEL_DIR, "cnn_classifier_history.pkl")
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)
    
    # Plot Training History
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_training_history.png"))
    plt.close()
else:
    print("Warning: Training history not found. Skipping history plots.")

print(f"Evaluation results saved to {RESULTS_DIR}")
