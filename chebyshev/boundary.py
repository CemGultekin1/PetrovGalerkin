import numpy.polynomial.chebyshev as cheb
import numpy as np
from .element import Degree

class Boundary(Degree):
    def __init__(self,degree:int) -> None:
        super().__init__(degree)
        self.degree = degree
        self.right = np.empty((self.degree,),dtype = float)
        self.left = np.empty((self.degree,),dtype = float)
    def fillup(self,):...
    
def flux_element(deg:int,right:bool = True):
    x = np.zeros((deg+1),dtype = float)
    x[deg] = 1
    if right:
        return -cheb.chebval(1,x)
    else:
        return cheb.chebval(-1,x)

def sum_value(deg:int,right:bool = True):
    x = np.zeros((deg+1),dtype = float)
    x[deg] = 1
    if right:
        return cheb.chebval(1,x)
    else:
        return cheb.chebval(-1,x)
    
class Flux(Boundary):
    def fillup(self,):
        for i in self.degree_index_product(1):
            self.right[i] =  flux_element(*i,right=True)
            self.left[i] =  flux_element(*i,right=False)
            
class Value(Boundary):
    def fillup(self,):
        for i in self.degree_index_product(1):
            self.right[i] =  sum_value(*i,right=True)
            self.left[i] =  sum_value(*i,right=False)


def degree_mat_multip(degvec:np.ndarray,bdrmat:np.ndarray):    
    bdrdeg = degvec.reshape([1,-1,1])*np.stack([bdrmat],axis = 1)
    bdrdeg = bdrdeg.reshape([bdrdeg.shape[0],-1])
    return bdrdeg