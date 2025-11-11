# NeuroLite — Lightweight Deep Learning Framework (From Scratch)

**NeuroLite** is a minimal deep learning framework implemented from first principles using **NumPy**. It features a custom **Tensor** class that builds dynamic computation graphs and performs **reverse-mode automatic differentiation**. This design is similar to micrograd, but extended to support matrix operations, broadcasting, and multi-layer neural networks.

The goal of this project is to make neural networks transparent and understandable at the lowest level.

---

## Features

- **Tensor Class** with:
  - Reverse‑mode autodiff (backpropagation)
  - Broadcasting support
  - Matrix multiplication (`@`)
  - Element‑wise operations (`+`, `-`, `*`, `/`)
  - Activation functions (`tanh`, more to come)

- - `Layer` and `MLP` classes for building neural networks
- Training support with gradient descent

---

## Computation Graph Overview

```
Tensor(data)
   │
   ├── Operations: +, -, *, /, @, tanh, sum, etc.
   │
   └── Produces new Tensors that reference previous nodes (_prev)
```

Calling `.backward()`:
1. Builds a topological order of nodes.
2. Propagates gradients backward through the graph.

---
---
## 🔥 Autodiff: The Core of NeuroLite

The **Tensor** class is the heart of this framework.  
Every operation creates a new `Tensor` that remembers **how it was created**, forming a **computation graph**.  
Calling `.backward()` performs **reverse-mode automatic differentiation** (the same technique used in PyTorch).

---

## Scalar Example

```python
from engine import Tensor

a = Tensor(2.0, label='a')
b = Tensor(3.0, label='b')

c = a * b
d = c + a
e = d.tanh()

e.backward()
print("Value e:", e.data) #Value e: 0.9999997749296758
print("Gradient wrt a:", a.grad) #Gradient wrt a: 1.8005623907413337e-06
print("Gradient wrt b:", b.grad) #Gradient wrt b: 9.002811953706669e-07
```

---
### Computation Graph
a ----*
      |→ (mul) → c ----*
b ----*               |
                      |→ (add) → d → tanh → e
a --------------------*
When .backward() is called on e, gradients flow backwards through this graph automatically.
## Matrix Example

```python
import numpy as np
from engine import Tensor

X = Tensor([[1., 2.], [3., 4.]])
W = Tensor([[0.5, -1.0], [2.0, 1.5]])
b = Tensor([1.0, -2.0])

Z = X @ W + b
A = Z.tanh()

loss = A.sum()
loss.backward()

print("X.grad:\n", X.grad) #X.grad:[[-0.9999666   1.50013361][-0.41997434  0.62996152]]
print("W.grad:\n", W.grad) #W.grad:[[6.68136707e-05 2.25992302e+00][1.33621275e-04 3.67989737e+00]]
print("b.grad:\n", b.grad) #b.grad:[6.68076047e-05 1.41997434e+00]
```

---

## XOR Example

```python
import numpy as np
from engine import Tensor, MLP, mse_loss, sgd_step

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

model = MLP(in_features=2, hidden_sizes=[4], out_features=1)

for step in range(2000):
    y_pred = model.forward(X)
    loss = mse_loss(y_pred, y)
    loss.backward()
    sgd_step(model.parameters(), lr=0.1)

print("Predictions:")
print(model.forward(X).data) #gives [[0.],[1.],[1.][0.]]
```

---

## Roadmap

| Feature             | Purpose                        |
|--------------------|--------------------------------|
| ReLU               | Avoid saturation issues        |
| Cross‑Entropy Loss  | Multi‑class classification     |
| Adam Optimizer      | Faster smoother convergence    |

---

## License
MIT
