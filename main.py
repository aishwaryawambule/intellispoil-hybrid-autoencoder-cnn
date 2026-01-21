# This script serves as the primary entry point for the IntelliSpoil CNN Classifier application.
# It sets up the necessary environment and executes the main classification logic.
import os
import sys

from src.convolutional_neural_network.cnn_model_inference import main

if __name__ == "__main__":
    """
    Entry point of the application.
    Executes the main function from the cnn_classifier module.
    """
    main()
