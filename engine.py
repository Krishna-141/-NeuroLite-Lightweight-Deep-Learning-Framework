import math
import numpy as np

class Tensor:

    def __init__(self,data,_children=(),_op='',label=''):
        self.data=np.array(data,dtype=np.float64)
        self._op=_op
        self._prev=set(_children)
        self.label=label
        self.grad =np.zeros_like(self.data)
        self._backward = lambda: None
        
    def __repr__(self):
        return f"{self.data}"
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Handle broadcasting by summing over broadcasted dimensions
      
            if out.grad.shape != self.data.shape:
                self_grad = np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(self.data.shape))), keepdims=True)
                self_grad = np.reshape(self_grad, self.data.shape)
            else:
                self_grad = out.grad
                
       
            if out.grad.shape != other.data.shape:
                other_grad = np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(other.data.shape))), keepdims=True)
                other_grad = np.reshape(other_grad, other.data.shape)
            else:
                other_grad = out.grad
                
            self.grad += self_grad
            other.grad += other_grad
            
        out._backward = _backward
        return out
         
    def __radd__(self, other):  # other + self
        return self + other     
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        
        def _backward():
            # Handle broadcasting for self
            if out.grad.shape != self.data.shape:
                self_grad = np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(self.data.shape))), keepdims=True)
                self_grad = np.reshape(self_grad, self.data.shape)
            else:
                self_grad = out.grad
                
            # Handle broadcasting for other  
            if out.grad.shape != other.data.shape:
                other_grad = np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - len(other.data.shape))), keepdims=True)
                other_grad = np.reshape(other_grad, other.data.shape)
            else:
                other_grad = out.grad
                
            self.grad += self_grad
            other.grad -= other_grad  # Note the minus sign here
            
        out._backward = _backward
        return out
    
    def __rsub__(self, other):  # other - self
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self
        
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
    
        def _backward():
            # Expand gradient to the original shape
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
    
        out._backward = _backward
        return out

    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Handle broadcasting for self
            if out.grad.shape != self.data.shape:
                self_grad = np.sum(out.grad * other.data, axis=tuple(range(len(out.grad.shape) - len(self.data.shape))), keepdims=True)
                self_grad = np.reshape(self_grad, self.data.shape)
            else:
                self_grad = out.grad * other.data
                
            # Handle broadcasting for other  
            if out.grad.shape != other.data.shape:
                other_grad = np.sum(out.grad * self.data, axis=tuple(range(len(out.grad.shape) - len(other.data.shape))), keepdims=True)
                other_grad = np.reshape(other_grad, other.data.shape)
            else:
                other_grad = out.grad * self.data
                
            self.grad += self_grad
            other.grad += other_grad
            
        out._backward = _backward
        return out
        
    def __rmul__(self, other):  # other * self
        return self * other
     
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        
        def _backward():
            # Handle broadcasting for self
            if out.grad.shape != self.data.shape:
                self_grad = np.sum(out.grad / other.data, axis=tuple(range(len(out.grad.shape) - len(self.data.shape))), keepdims=True)
                self_grad = np.reshape(self_grad, self.data.shape)
            else:
                self_grad = out.grad / other.data
                
            # Handle broadcasting for other  
            if out.grad.shape != other.data.shape:
                other_grad = np.sum(-self.data * out.grad / (other.data ** 2), axis=tuple(range(len(out.grad.shape) - len(other.data.shape))), keepdims=True)
                other_grad = np.reshape(other_grad, other.data.shape)
            else:
                other_grad = -self.data * out.grad / (other.data ** 2)
                
            self.grad += self_grad
            other.grad += other_grad
            
        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), 'matmul')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
        
    def __rtruediv__(self, other):  
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    def __pow__(self, power):
        if isinstance(power, (int, float)):
            out = Tensor(self.data ** power, (self,), f'**{power}')
            
            def _backward():
                self.grad += (power * (self.data ** (power - 1))) * out.grad
                
            out._backward = _backward
            return out
        else:
            raise TensorError("Power must be scalar")
    
    def __neg__(self):  # -self
        return self * -1    
        
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out


    def backward(self):
        topo=[]
        visited=set()
        def dfs(node):
            if node not in visited: visited.add(node)
            for child in node._prev:
                dfs(child)
            topo.append(node)
        dfs(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()