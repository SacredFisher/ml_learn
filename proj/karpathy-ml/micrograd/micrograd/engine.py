#Value class for micrograd
import math

class Value:

    def __init__(self, data, _inputs = (), _op = '', _label = ''):

        #value, and grad
        self.data = data
        self.grad = None

        #internal vars
        self._backward = lambda: None  #backward function, derivative
        self._inputs = set(_inputs)    #the inputs to the function
        self._op = _op                 #the operation creating the value
        self._label = _label           #the label of the value, for viz

    def __repr__(self): 
        return f'Value({self.data})'

    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data , _inputs = (self, other), _op = '+')
        
        def _backward(): 
            #c = a + b 
            #dc/da = (da/da + db/da) = 1
            #v = f(c), then dv/da = dc/da * dv/dc
            self.grad = 1 * out.grad #local derivative is 1, * incoming derivative by chain rule
            out.grad = 1 * out.grad 
        
        out._backward = _backward
        
        return out
    
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _inputs = (self, other), _op = '*')
        
        def _backward(): 
            #c = a*b
            #dc/da = (d(a*b)/da) = b
            #v = f(c), then dv/da = dc/da * dv/dc
            self.grad = other.grad * out.grad  #local derivative is other.grad, * incoming derivative by chain rule
            other.grad = self.grad * out.grad 

        out._backward = _backward 

        return out 
    
    def __tanh__(self): 
        t = math.tanh(self.data)
        out = Value(t, _inputs = (self, ), _op = 'tanh()')

        def _backward(): 
            self.grad = (1 - t**2) * out.grad  

        out._backward = _backward 

        return out 

if __name__ == '__main__': 
    a = Value(1, _label ='a')
    b = Value(2, _label ='b')
    c = a + b 
    print(c)

