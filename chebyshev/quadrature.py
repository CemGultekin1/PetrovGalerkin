from .element  import Degree
import numpy.polynomial.chebyshev as cheb
import numpy as np

def quadfun(*degrees:int, first_derivative:bool = False):
    y = np.ones([1],dtype = float)
    for i,deg in enumerate(degrees):
        x = np.zeros((deg+1),dtype = float)
        x[deg] = 1
        if i == 0 and first_derivative:
            x = cheb.chebder(x)
        y = cheb.chebmul(y,x)
    yint = cheb.chebint(y)
    return cheb.chebval(1,yint) - cheb.chebval(-1,yint)
    

class QuadratureTensor(Degree):
    def __init__(self,degree:int) -> None:
        super().__init__(degree)
        self.tri_quads = np.empty((self.degree,self.degree,self.degree),dtype = float)
        self.der_dub_quads = np.empty((self.degree,self.degree,),dtype = float)
        self.dub_quads = np.empty((self.degree,self.degree,),dtype = float)
        self.filledup = False
    def fillup(self,):
        for i,j,k in self.degree_index_product(3):
            self.tri_quads[i,j,k] = -quadfun(i,j,k)
        for i,j in self.degree_index_product(2):
            self.dub_quads[i,j] = quadfun(i,j)
            self.der_dub_quads[i,j] = -quadfun(i,j,first_derivative=True)
        self.filledup = True