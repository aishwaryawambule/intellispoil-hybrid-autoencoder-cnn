import tensorflow as tf
import os
import sys

# Define the base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data/train_test_dataset")

IMG_SIZE = 450
BATCH_SIZE = 40

def load_binary_dataset(directory):
    """
    Loads a dataset from a directory and maps it to a binary classification task (Healthy vs Rotten).
    
    Args:
        directory (str): Path to the dataset directory.
        
    Returns:
        tf.data.Dataset: A dataset yielding (image, label) pairs.
    """
    # 1. Load with label_mode='int' to get indices
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode='int', 
        # class_names=['freshapples', 'rottenapples'], # Removed hardcoded class names to allow flexibility
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # 2. Create a mapping based on folder names
    # Folders starting with 'fresh' -> 0 (Healthy)
    # Folders starting with 'rotten' -> 1 (Rotten)
    class_names = ds.class_names
    label_map = [0 if name.lower().startswith('fresh') else 1 for name in class_names]
    label_map_tf = tf.constant(label_map, dtype=tf.int32)

    print(f"Mapping {len(class_names)} folders to 2 classes:")
    for i, name in enumerate(class_names):
        print(f"  Folder '{name}' -> {'Healthy' if label_map[i] == 0 else 'Rotten'}")

    # 3. Apply the mapping function to the dataset
    def map_to_binary(img, label):
        # Use tf.gather to look up the binary label (0 or 1) using the original index
        binary_label = tf.gather(label_map_tf, label)
        return img, tf.cast(binary_label, tf.float32)

    return ds.map(map_to_binary)

def load_dataset(directory, fruit_name):
    """
    Loads a dataset for a specific fruit (e.g., 'banana') and maps it to binary labels.
    
    Args:
        directory (str): Path to the dataset directory.
        fruit_name (str): Name of the fruit to filter by (e.g., 'banana').
        
    Returns:
        tf.data.Dataset: A dataset yielding (image, label) pairs for the specific fruit.
    """
    # Load only folders for a specific fruit
    class_names = [f'fresh{fruit_name}', f'rotten{fruit_name}']
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode='int', 
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    label_map = [0, 1] # fresh -> 0, rotten -> 1
    label_map_tf = tf.constant(label_map, dtype=tf.int32)

    def map_to_binary(img, label):
        binary_label = tf.gather(label_map_tf, label)
        return img, tf.cast(binary_label, tf.float32)

    return ds.map(map_to_binary)

print("Loading Train Dataset...")
train_ds = load_binary_dataset(os.path.join(DATASET_PATH, "Train"))

print("\nLoading Test Dataset...")
test_ds = load_binary_dataset(os.path.join(DATASET_PATH, "Test"))

# Normalization and Prefetching
def normalize(img, label):
    """
    Normalizes image pixel values to the range [0, 1].
    """
    return tf.cast(img, tf.float32) / 255.0, label

train_ds = train_ds.map(normalize).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(normalize).prefetch(tf.data.AUTOTUNE)

# Dataset for Autoencoder (X, X) - Autoencoders learn to reconstruct the input
train_ds_aen = train_ds.map(lambda x, y: (x, x))
test_ds_aen = test_ds.map(lambda x, y: (x, x))