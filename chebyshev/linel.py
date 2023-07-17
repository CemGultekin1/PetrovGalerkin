import numpy as np
from .element import ChebyshevElement

        
class Dimensionalizer(ChebyshevElement):
    def lhs_element(self,featval:np.ndarray,):
        dim,degree = self.dim,self.degree
        x = np.eye(dim).reshape([1,1,dim,dim])*featval.flatten()
        x = x.reshape([degree,degree,dim,dim]).transpose([0,2,1,3]).reshape([degree*dim,degree*dim])
        return x
    def rhs_element(self,featval:np.ndarray,):
        return featval
    def __call__(self,featval:np.ndarray,)->np.ndarray:
        # self.infer_degree(featval)
        if featval.size == self.dim*self.degree:
            return self.rhs_element(featval)
        else:
            return self.lhs_element(featval)
        
class LinearElement(ChebyshevElement):
    def __init__(self, dim: int, degree: int) -> None:
        super().__init__(dim, degree)
        self.dimensionalized = False
    def dimensionalize(self,):
        if self.dimensionalized:
            return
        dimensionalizer = Dimensionalizer(self.dim,self.degree)
        for featkey in self.__dict__.keys():
            if '_element' not in featkey:
                continue
            featval :np.ndarray= self.__dict__[featkey]
            self.__dict__[featkey] = dimensionalizer(featval)
        self.dimensionalized = True