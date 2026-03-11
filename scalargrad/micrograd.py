import math

class Value:
    def __init__(self,data,prev=()):
        self.data = data
        self.prev = set(prev)
        self.grad = 0.
        self._backward = lambda : None
    
    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self,other):
        other =  other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data, (self,other))
        def backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = backward
        return out

    def __mul__(self,other):
        other =  other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)), 'only supporting int/float now'
        out = Value(self.data ** other, (self,))

        def backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = backward
        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data),(self,))
        def backward():
            t = out.data
            self.grad += (1-t**2) * out.grad
        out._backward = backward
        return out
    
    def relu(self):
        out = Value(self.data if self.data>0 else 0, (self,))
        def backward():
            if self.data > 0:
                self.grad += 1 * out.grad
            else:
                self.grad += 0 * out.grad
        out._backward = backward
        return out
    
    def __truediv__(self,other):
        return self * other ** -1

    def __neg__(self):
        return self * -1

    def __sub__(self,other):
        return self + (-other)
    def abs(self):
        sign = self.data > 0
        out = Value(self.data * sign,(self,))
        def backward():
            self.grad += sign * out.grad
        out._backward = backward
        return out

    
    def __radd__(self,other):
        return self + other
    def __rsub__(self,other):
        return other + (- self)
    def __rtruediv__(self,other):
        return other * self**-1
    def __rmul__(self,other):
        return self * other    
        
    def backward(self):
        self.grad = 1.
        visited = set()
        topo = []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v.prev:
                    build_topo(prev)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo):
            v._backward()