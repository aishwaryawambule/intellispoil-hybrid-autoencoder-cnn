# IntelliSpoil: Fruit Freshness Classification

IntelliSpoil is a Deep Learning project designed to classify fruits and Vegetables as either "Fresh" or "Rotten". It leverages a **Convolutional Autoencoder (CAE)** for unsupervised feature extraction and a **Convolutional Neural Network (CNN)** for binary classification.

## Project Overview

The project consists of two main stages:
1.  **Feature Extraction (Autoencoder):** A Convolutional Autoencoder is trained to reconstruct input images. The encoder part of this model learns a compressed, latent representation of the images, capturing essential features.
2.  **Classification (CNN):** A CNN classifier uses the pre-trained encoder as a feature extractor. It takes the latent features produced by the encoder and passes them through a classification head to determine if the fruit is fresh or rotten.

## Directory Structure

```
.
├── main.py                     # Entry point for prediction
├── requirements.txt            # Python dependencies
├── src/
│   ├── convolutional_autoencoder/
│   │   ├── con_aen.py          # Script to train the Autoencoder
│   │   ├── con_aen_encoder.py  # Encoder architecture definition
│   │   └── con_aen_decoder.py  # Decoder architecture definition
│   ├── convolutional_neural_network/
│   │   ├── cnn_classifier.py   # Script to train the Classifier
│   │   ├── cnn_model_prediction.py # Script for running predictions
│   │   └── aen_reconstruction.py # Helper to load data for classifier
│   ├── data_pipeline.py        # Data loading and preprocessing logic
│   ├── data/              
│   │   ├── train_test_dataset/ # Dataset for training and testing
│   │   ├── prediction_use_dataset/ # Dataset for prediction
│   └── model/                  # Directory where trained models are saved
```

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Autoencoder

First, train the autoencoder to learn feature representations from the data.

```bash
uv run src/convolutional_autoencoder/con_aen.py
```

This will save the encoder, decoder, and full autoencoder models in the `src/model/` directory.

### 2. Train the Classifier

Once the autoencoder is trained, train the classifier. This script loads the pre-trained encoder, freezes its weights, and trains a classification head on top of it.

```bash
uv run src/convolutional_neural_network/cnn_classifier.py
```

This will save the final classifier model in `src/model/cnn_classifier_model.keras`.

### 3. Run Predictions

To run predictions on new data (e.g., the banana dataset):

```bash
uv run main.py
```

Or directly run the prediction script:

```bash
uv run src/convolutional_neural_network/cnn_model_prediction.py
```

## Dataset

-   **train_test_dataset**: Used for training and testing the Autoencoder and Classifier
-   **prediction_use_dataset**: Used for predicting the model's generalization

## Technologies Used

-   **TensorFlow/Keras**: Deep Learning framework.
-   **Python**: Programming language.
-   **NumPy & Matplotlib**: Data manipulation and visualization.
