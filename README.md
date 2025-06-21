# Optimized CNN and CRNN for Advanced Classification and Regression

This repository contains the implementation for my third major project, "Optimized CNN and CRNN for Advanced Classification and Regression," developed as part of my application for the Computer Vision role at Awiros.

While the project title mentions CRNN, due to time constraints, the focus of this immediate implementation is on the advanced CNN components for multi-task learning and image processing, laying the groundwork for future CRNN extensions for sequential data.

## Project Components:

### 1. Optimized CNNs for Multi-MNIST Multi-Label Classification/Regression
* **Description:** This component focuses on building and optimizing a Convolutional Neural Network (CNN) to perform two related tasks simultaneously on custom-generated Multi-MNIST images: multi-label classification and regression. The model's classification head identifies both digits present in the image, while its regression head predicts their approximate x-coordinates.
* **Key Aspects:**
    * **Custom Multi-MNIST Dataset:** Generates images with two MNIST digits side-by-side, providing both classification (digit identities) and regression (digit positions) labels.
    * **Multi-Task CNN Architecture:** A shared CNN backbone branches into separate fully connected heads for classification (predicting two digits) and regression (predicting two x-coordinates).
    * **Combined Loss Function:** Uses `CrossEntropyLoss` for classification and `MSELoss` for regression, allowing for joint optimization.
    * **Hyperparameter Tuning & Regularization:** Employs the Adam optimizer and includes Dropout layers for robustness.
    * **Feature Map Visualization:** Tools are integrated to visualize the activations of various convolutional layers, aiding in understanding the network's learning process.
* **Code:** See `multi_mnist_cnn.ipynb`.

### 2. CNN Autoencoder on Fashion-MNIST for Image Reconstruction and Dimensionality Reduction
* **Description:** This part implements a Convolutional Autoencoder (CAE) on the Fashion-MNIST dataset, primarily for image reconstruction and learning efficient, lower-dimensional representations of images.
* **Key Aspects:**
    * **Autoencoder Architecture:** Comprises an encoder (convolutional layers for compression) and a decoder (transpose convolutional layers for reconstruction).
    * **Image Reconstruction:** Trained using `MSELoss` to minimize the difference between original and reconstructed images.
    * **Dimensionality Reduction:** The bottleneck layer of the autoencoder provides a compact, low-dimensional representation (e.g., 64 features for a 784-pixel image), demonstrating effective data compression.
    * **Enhancing Classification Tasks:** The learned latent features can serve as powerful inputs for downstream classification models, potentially improving performance and training efficiency.
* **Code:** See `fashion_mnist_autoencoder.ipynb` .

## Setup and Usage:
1.  Clone this repository: `git clone https://github.com/YourUsername/Optimized-CNN-CRNN-Projects.git`
2.  Open the `.ipynb` notebooks in Google Colab or Jupyter environment.
3.  Install necessary libraries (e.g., `pip install torch torchvision matplotlib numpy pillow`).
4.  Run the cells sequentially.

---
