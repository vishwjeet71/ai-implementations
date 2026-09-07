# Backpropagation

Backpropagation is an algorithm used to train deep learning models.
It calculates gradients for the weights using the chain rule, and then the optimizer updates the weights using gradient descent.

Weight update formula:

$$W_{new}=W-(Learning\ Rate\times dW)$$

## How does it work

In this repo, we will understand how PyTorch handles backpropagation.
We will not dive deep into the math behind gradient calculation.

In PyTorch, training mainly happens in 2 steps.

### 1. `loss.backward()`

When we call `loss.backward()`, PyTorch calculates gradients for:

* Inputs
* Weights

### Operation 1: Backpropagation for Inputs

Formula:

$$dX=dY\times W^T$$

This formula calculates how much error should be passed to the previous layer.

* `dY` → error coming from the next layer
* `W` → weights of the current layer
* `dX` → error sent to the previous layer

### Operation 2: Backpropagation for Weights

Formula:

$$dW=X^T\times dY$$

This formula calculates gradients for the weights of the current layer.

* `X` → input from the previous layer
* `dY` → error coming from the next layer
* `dW` → gradient of weights

## Example

Suppose we have 3 layers:

Layer 1 → Layer 2 → Layer 3 → Output

If the final output is wrong, the error first reaches Layer 3.
We call this error `dY₃`.

### At Layer 3

* Layer 3 receives `dY₃`

* It calculates weight gradients:

  `dW₃ = X₃ᵀ × dY₃`

* Then it calculates the error for Layer 2:

  `dX₃ = dY₃ × W₃ᵀ`

* This `dX₃` becomes `dY₂` for Layer 2

### At Layer 2

* Layer 2 receives `dY₂`

* It calculates:

  `dW₂ = X₂ᵀ × dY₂`

* Then it calculates:

  `dX₂ = dY₂ × W₂ᵀ`

* This error is passed to Layer 1

And this process continues until the first layer.

```
PyTorch traverses the layers in reverse order, simultaneously performing both computations at each layer.
```

### 2. `optimizer.step()`

After gradients are calculated, the optimizer updates the weights using those gradients.

Backpropagation calculates gradients.
The optimizer uses those gradients to improve the model.
