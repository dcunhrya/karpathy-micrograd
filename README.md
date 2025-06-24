# My Custom Micrograd: A Tiny Autograd Engine (Inspired by Andrej Karpathy)

This repository contains a minimalist automatic differentiation (autograd) engine, built from scratch in Python, following the foundational concepts taught by Andrej Karpathy in his "micrograd" series. The goal of this project is to provide a deep understanding of how backpropagation works at a fundamental level, by manually implementing the core mechanics of a neural network library.

## 🌟 Features

* **Automatic Differentiation:** Implements `Value` objects that track operations and build a computation graph, allowing for automatic gradient calculation (`.backward()`).
* **Neural Network Building Blocks:** Includes basic neural network components like `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron).
* **Simple Training Loop:** Demonstrates how to define a loss function, perform backpropagation, and update parameters using gradient descent for binary classification tasks (e.g., `make_moons` dataset).
* **Visualization:** Integration with `graphviz` to visualize the computation graph and gradients (as shown in `micrograd_scratch.ipynb`).

## 📁 Project Structure
dcunhrya-karpathy-micrograd/
* demo.ipynb               # Main notebook demonstrating training on make_moons
* micrograd_scratch.ipynb  # Core concepts of Value and NN components (for development)
* micrograd/
  * engine.py            # Defines the Value class and core autograd operations
  * nn.py                # Defines Neuron, Layer, MLP classes

 ## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/dcunhrya-karpathy-micrograd.git](https://github.com/your-username/dcunhrya-karpathy-micrograd.git)
    cd dcunhrya-karpathy-micrograd
    ```
2.  **Install dependencies:**
    This project requires `numpy`, `matplotlib`, `scikit-learn`, and `graphviz`.
    ```bash
    pip install numpy matplotlib scikit-learn graphviz
    ```
    * **Note on Graphviz:** For graph visualization, you might also need to install the Graphviz system package.
        * On Debian/Ubuntu: `sudo apt-get install graphviz`
        * On macOS (with Homebrew): `brew install graphviz`
        * On Windows: Download and install from [graphviz.org](https://graphviz.org/download/) and add it to your PATH.

## 💡 Usage

The primary demonstration of this `micrograd` implementation is in `demo.ipynb`.

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook demo.ipynb
    ```
2.  **Run cells sequentially:** The notebook walks through data loading, model definition, the loss function, and the training loop for classifying the `make_moons` dataset.

The `micrograd_scratch.ipynb` notebook provides a more detailed, step-by-step exploration of the `Value` class and its operations, which is excellent for understanding the inner workings of the autograd engine.

## 🙏 Acknowledgements

This project is directly inspired by and based on **Andrej Karpathy's "micrograd"** series and repository. 

* [Andrej Karpathy's micrograd repository](https://github.com/karpathy/micrograd)
* [Andrej Karpathy's YouTube series](https://www.youtube.com/watch?v=VMj-3S1tku0) (specifically "The spelled-out intro to neural networks and backpropagation: building micrograd")
