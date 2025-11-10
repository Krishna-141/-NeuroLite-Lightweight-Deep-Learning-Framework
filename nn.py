from engine import Tensor
import numpy as np

class Layer:
    def __init__(self, in_features, out_features, activation="tanh"):
        # Xavier/Glorot-like init
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)))
        self.b = Tensor(np.zeros(out_features))
        self.activation = activation

    def forward(self, X): # X: (batch, in_features)
        if not isinstance(X, Tensor):
            X = Tensor(X)

        Z = X @ self.W + self.b  # (batch, out_features)
        if self.activation == "tanh":
            return Z.tanh()
        if self.activation is None:
            return Z
        raise ValueError("Unsupported activation")

    def parameters(self):
        return [self.W, self.b]

# ----- MLP -----
class MLP:
    def __init__(self, in_features, hidden_sizes, out_features, last_activation=None):
        sizes = [in_features] + hidden_sizes + [out_features]
        self.layers = []
        for i in range(len(sizes) - 2):
            self.layers.append(Layer(sizes[i], sizes[i+1], activation="tanh"))
        self.layers.append(Layer(sizes[-2], sizes[-1], activation=last_activation))

    def forward(self, X):
        if not isinstance(X, Tensor):
            out = Tensor(X)
        else:
            out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out # shape: (batch, out_features)

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params