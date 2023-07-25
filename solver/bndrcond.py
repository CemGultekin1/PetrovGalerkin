from dataclasses import dataclass
import logging
from typing import   Tuple
from chebyshev import  GridwiseChebyshev,NumericFunType,ListOfFuns
import numpy as np

def is_square(x:np.ndarray):
    if x.ndim != 2:
        return False
    if x.shape[0] != x.shape[1]:
        return False
    return True
def same_shape(x1:np.ndarray,x2:np.ndarray):
    return x1.shape == x2.shape
class BoundaryCondition:
    B0:np.ndarray
    B1:np.ndarray
    c:np.ndarray
    def __init__(self,b0:np.ndarray,b1:np.ndarray,c:np.ndarray) -> None:
        self.B0 = b0
        self.B1 = b1
        self.c = c
        try:
            if (not is_square(b0) ) or (not is_square(b1)) or (not same_shape(b0,b1)):
                raise Exception
            if b0.shape[0] != c.shape[0]:
                raise Exception
        except:
            logging.error(f'Boundary condition dimensions are not consistent: {b0.shape},{b1.shape},{c.shape}' )
            raise Exception

    def adjoint(self,)->'BoundaryCondition':...
    @property
    def dim(self,):
        return self.B0.shape[0]




        
        