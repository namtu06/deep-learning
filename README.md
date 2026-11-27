# Deep Learning Coursework

Coursework for **DATA.ML.200: Deep Learning**. This repository follows a six-week progression through core neural-network architectures, from manually deriving gradients to generative models.

## Weekly projects

| Week | Project | What you'll explore |
| --- | --- | --- |
| 1 | **Gradient descent and multilayer perceptrons** | Derive the mathematics behind gradient descent, work through backpropagation, implement gradients with NumPy, and build a multilayer perceptron. See the [pen-and-paper exercises](1_multilayer_perceptrons/ex1_01_pen_and_paper.ipynb), [NumPy gradient implementation](1_multilayer_perceptrons/ex1_02_numpy_gradient.ipynb), and [PyTorch regression exercise](1_multilayer_perceptrons/ex1_03_torch_regression.ipynb). |
| 2 | **Image denoising with a U-Net CNN** | Train a convolutional neural network with a U-Net-style architecture to remove noise from images. See the [notebook](2_convolutional_neural_networks/ex2.ipynb), [MiniUNet diagram](2_convolutional_neural_networks/MiniUNet.png), and [denoising example](2_convolutional_neural_networks/denoising_example.PNG). |
| 3 | **CNN sound classification and hyperparameter tuning** | Fine-tune a CNN for sound classification by experimenting with training hyperparameters. See the [Week 3 notebook](3_exploring_hyperparameters/ex3.ipynb) and its [helper utilities](3_exploring_hyperparameters/ex3_helpers.py). |
| 4 | **Next-character prediction with an RNN** | Train a recurrent neural network to predict the next character in text from *Alice's Adventures in Wonderland*. See the [notebook](4_recurrent_nn/ex4.ipynb) and [text data](4_recurrent_nn/alice.txt). |
| 5 | **French-to-English machine translation with a Transformer** | Implement a Transformer for sequence-to-sequence translation from French to English. See the [notebook](5_transformers/ex5.ipynb) and [helper utilities](5_transformers/ex5_helpers.py). |
| 6 | **Handwritten-digit generation with a GAN** | Build a generative adversarial network that learns to generate handwritten-digit images from random noise. The Week 6 project is part of the course plan, but its files are not currently in this repository. |

## Repository structure

Each numbered directory contains the notebooks and supporting materials for that part of the coursework. Most exercises are delivered as Jupyter notebooks; consult each notebook for its implementation and exercise-specific details.
