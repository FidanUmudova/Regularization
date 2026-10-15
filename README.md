# Deep Learning Regularization and Dropout

This repository contains practical implementations of **L2 Regularization** and **Dropout** in a 3-layer neural network using NumPy. Completed as part of deep learning coursework, this project focuses on mitigating overfitting and improving model generalization.

##  Implemented Methods

1. **L2 Regularization (`compute_cost_with_regularization` & `backward_propagation_with_regularization`)**
   * Adds an L2 penalty term ($\frac{\lambda}{2m} \sum W^2$) to the cross-entropy cost function.
   * Updates backpropagation gradients (`dW`) to penalize large weights and prevent overfitting.

2. **Inverted Dropout (`forward_propagation_with_dropout` & `backward_propagation_with_dropout`)**
   * Randomly shuts down neurons in hidden layers with a probability of `1 - keep_prob` during training.
   * Scales the remaining active neurons by `keep_prob` (Inverted Dropout technique) to maintain the expected value of activations and ensure smooth transition during testing.

##  Tech Stack
* **Python**
* **NumPy**
* **Deep Learning Fundamentals**
