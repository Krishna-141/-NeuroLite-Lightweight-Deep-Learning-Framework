# NeuroLite — Lightweight Deep Learning Framework (From Scratch)

NeuroLite is a minimal deep learning framework implemented entirely from first principles using **NumPy**.  
It includes a custom **Tensor** class that builds dynamic computation graphs and performs **backpropagation** automatically — similar in spirit to micrograd, but extended to support matrix operations, broadcasting, and multi-layer neural networks.

The goal of this project is to make neural networks **transparent** and **understandable** at the lowest level.

---

## Features

- `Tensor` class with:
  - Automatic differentiation (reverse-mode autodiff)
  - Support for broadcasting in gradients
  - Matrix multiplication (`@`)
  - Elementwise operations (`+`, `-`, `*`, `/`)
  - Activation functions (e.g., `tanh`)
- `Layer` and `MLP` classes for building neural networks
- Training with gradient descent

---

## Tensor Computation Graph Overview

Tensor(data)
│
├── operations: +, -, *, /, @, tanh, sum, etc.
│
└── produces new Tensors that reference previous ones (_prev)


During `.backward()`, NeuroLite:
1. Builds a **topological ordering** of the graph
2. Propagates gradients backwards through each operation

---
## 🔥 Autodiff: The Core of NeuroLite

The **Tensor** class is the heart of this framework.  
Every operation creates a new `Tensor` that remembers **how it was created**, forming a **computation graph**.  
Calling `.backward()` performs **reverse-mode automatic differentiation** (the same technique used in PyTorch).

---

### ✅ Example 1 — Scalar Autodiff (Easy to See)

```python
from engine import Tensor

a = Tensor(2.0, label='a')
b = Tensor(3.0, label='b')

c = a * b              # c = 2 * 3
d = c + a              # d = c + a
e = d.tanh()           # apply activation

e.backward()

print("Value e:", e.data) #Value e: 0.9999997749296758
print("Gradient wrt a:", a.grad) #Gradient wrt a: 1.8005623907413337e-06
print("Gradient wrt b:", b.grad) #Gradient wrt b: 9.002811953706669e-07

### Computation Graph
a ----*
      |→ (mul) → c ----*
b ----*               |
                      |→ (add) → d → tanh → e
a --------------------*
When .backward() is called on e, gradients flow backwards through this graph automatically.
    
### ✅ Example 2 — Matrix Autodiff (Neural Network Style)
```python
import numpy as np
from engine import Tensor

X = Tensor([[1., 2.],
            [3., 4.]])        # (2,2)

W = Tensor([[0.5, -1.0],
            [2.0,  1.5]])     # (2,2)

b = Tensor([1.0, -2.0])       # (2,)

Z = X @ W + b                 # matrix multiply + broadcast
A = Z.tanh()                  # activation

loss = A.sum()                # scalar needed for backprop
loss.backward()

print("X.grad:\n", X.grad) #X.grad:[[-0.9999666   1.50013361][-0.41997434  0.62996152]]
print("W.grad:\n", W.grad) #W.grad:[[6.68136707e-05 2.25992302e+00][1.33621275e-04 3.67989737e+00]]
print("b.grad:\n", b.grad) #b.grad:[6.68076047e-05 1.41997434e+00]

## Example: Solving XOR

```python
import numpy as np

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float64)

y = np.array([[0],[1],[1],[0]], dtype=np.float64)

# Create model
model = MLP(in_features=2, hidden_sizes=[4], out_features=1, last_activation=None)

# Loss + Optimizer
for step in range(2000):
    y_pred = model.forward(X)
    loss = mse_loss(y_pred, y)
    loss.backward()
    sgd_step(model.parameters(), lr=0.1)

# Check learned outputs
print("Predictions:")
print(model.forward(X).data) #gives [[0.],[1.],[1.][0.]]

### Next Extensions
| Feature                 | Purpose                      |
| ----------------------- | ---------------------------- |
| ReLU                    | Avoid tanh saturation        |
| sigmoid + Cross Entropy | Multi-class classification   |
| Adam optimizer          | Faster and smoother training |


