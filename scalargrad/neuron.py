from micrograd import Value
import random
import math

class Neuron:
    def __init__(self,fan_in):
        self.weight = [Value(random.random() / math.sqrt(fan_in)) for _ in range(fan_in)]
        self.bias = Value(random.random())

    def __call__(self,x):
        return sum((xi * wi for xi,wi in zip(x,self.weight)),self.bias)
    
    def parameters(self):
        return [w for w in self.weight] + [self.bias]

class Layer:
    def __init__(self,fan_in,fan_out):
        self.neurons = [Neuron(fan_in) for _ in range(fan_out)]
    
    def __call__(self,x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self,fan_in,out):
        d = [fan_in] + out
        self.layers = [Layer(d[i],d[i+1]) for i in range(len(d)-1)]
    
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = [xi.tanh() for xi in x]
        x = self.layers[-1](x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
