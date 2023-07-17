from .element  import Degree
import numpy.polynomial.chebyshev as cheb
import numpy as np
from .interval import ChebyshevInterval
def quadrature(*degrees:int, first_derivative:bool = False):
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
    def fillup(self,):
        for i,j,k in self.degree_index_product(3):
            self.tri_quads[i,j,k] = quadrature(i,j,k)
        for i,j in self.degree_index_product(2):
            self.dub_quads[i,j] = quadrature(i,j)
            self.der_dub_quads[i,j] = quadrature(i,j,first_derivative=True)
    
    
class InteriorElement:
    mat_element:np.ndarray
    rhs_element:np.ndarray
    der_quads:np.ndarray
    def __init__(self,der_quads:np.ndarray,mat_element:np.ndarray,rhs_element:np.ndarray) -> None:
        self.der_quads = der_quads
        self.mat_element = mat_element        
        self.rhs_element = rhs_element            
        
class InteriorElementFactory(QuadratureTensor):
    def tri_multip(self,coeff:np.ndarray,degree:int)->np.ndarray:
        return np.dot(coeff,self.tri_quads[:degree,:degree,:degree]) # (galerkin elements) x (dimensions)
    def dub_multip(self,coeff:np.ndarray,degree:int)->np.ndarray:
        return np.dot(coeff,self.dub_quads[:degree,:degree])
    def generate_element(self,matfun:ChebyshevInterval,rhsfun:ChebyshevInterval):
        deg = matfun.degree
        matel = self.tri_multip(matfun.coeffs,deg)*matfun.h
        rhsel = self.dub_multip(rhsfun.coeffs,deg)*rhsfun.h
        der_quads = self.der_dub_quads[:deg,:deg]
        return InteriorElement(der_quads,matel,rhsel)