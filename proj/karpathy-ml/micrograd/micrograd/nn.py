from micrograd.engine import Value 
import random
import math

class Module: 
    def zero_grad(self): 
        for p in self.parameters():
            p.grad = 0 

    def parameters():
        return [] 


class Neuron(Module):
    def __init__(self, fan_in, non_lin = 'tanh'): 
        self.weights = [Value(random.uniform(-1,1)) for _ in range(fan_in)]
        self.b = Value(random.uniform(-1,1))
        self.non_lin = non_lin

    def __call__(self, xin): 
        if self.non_lin == 'tanh': 
            out = math.tanh(sum(w*x for w, x in zip(self.weights, xin)) + self.b)
        elif self.non_lin == 'relu': 
            out = (sum(w*x for w, x in zip(self.weights, xin)) + self.b).relu()
        elif self.non_lin == 'none': 
            out = sum(w*x for w, x in zip(self.weights, xin)) + self.b
        else: 
            raise ValueError('Neuron.non_lin is not one of "tanh", "relu", or "none". please check and try again.')
        
    def parameters(self): 
        return self.weights + [self.b]
    
    def __repr__(self):
        if self.non_lin == 'tanh':
            name = 'Tanh'
        elif self.non_lin == 'relu': 
            name = 'Relu'
        elif self.non_lin == 'none': 
            name = 'Linear'
        else: 
            raise ValueError('Neuron.non_lin is not one of "tanh", "relu", or "none". please check and try again.')
        
        return f'{name} Neuron({len(self.weights)})'