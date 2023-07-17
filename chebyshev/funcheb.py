import numpy as np
from typing import Tuple
import numpy.polynomial.chebyshev as cheb

# class MultiValuedInterpolationDecorator:
#     def __init__(self,_interpolator):
#         self._interpolator = _interpolator
#     def __call__(self, _fun,degree:int,shape:Tuple[int,...]) -> np.ndarray:

class CallMemory:
    def __init__(self,fun) -> None:
        self.fun = fun
        self.memory = []
        self.last_x = -np.inf
        self.last_i = 0
        self.shape = ()
        self.shape_interpreted = False
    def recognize_shape(self,output:np.ndarray):
        self.shape = output.shape
        self.shape_interpreted = True
    def __call__(self,x:float):
        if not self.shape_interpreted:
            val = self.fun(x)
            self.recognize_shape(val)
            self.memory.append(val)
        if x > self.last_x: # still continuing on the same memory
            

def foo(x:float):
    return [x,x**2]