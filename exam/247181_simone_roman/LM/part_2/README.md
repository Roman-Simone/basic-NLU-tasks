# Language Model Project


## Scripts and directory

### 1. `main_manual_avsgd.py`

This script implements the language model with manual Average Stochastic Gradient Descent. The manual version calculates the moving average of the model weights during training and mantain SGD optimizer.

### 2. `main_automatic_avsgd.py`

This script implements the language model using the automatic AvSGD available in PyTorch. The automatic version leverages PyTorch's built-in functionalities to switch from standard SGD to AvSGD during training.

### 3. `extra`

The folder `extra` contains details on how each model was trained, along with a graph of the loss. Additionally, there is a script `test_model.py` provided to test the various models.
