from dataclasses import dataclass
import numpy as np
from chebyshev import ChebyshevInterval,QuadratureTensor
    

class InteriorElement:
    mat_element:np.ndarray # deg' x deg x (dim x dim)
    rhs_element:np.ndarray # deg' x dim
    der_quads:np.ndarray # deg' x deg
    def __init__(self,der_quads:np.ndarray,mat_element:np.ndarray,rhs_element:np.ndarray) -> None:
        self.der_quads = der_quads
        self.mat_element = mat_element        
        self.rhs_element = rhs_element    
    def to_matrix_form(self,):
        dim = self.rhs_element.shape[1]
        deg = self.mat_element.shape[1]        
        # takes the transpose for adjoint
        mat = self.mat_element.reshape([-1,deg,dim,dim]).transpose((0,3,1,2)).reshape([-1,deg*dim]) 
        rhs = self.rhs_element.flatten()
        der = self.der_quads.reshape([-1,1])*np.eye(dim).reshape([1,-1])
        der = der.reshape([-1,deg,dim,dim]).transpose((0,2,1,3)).reshape([-1,deg*dim])
        return InteriorElementMatrices(mat + der, rhs)


    
@dataclass
class InteriorElementMatrices:
    mat:np.ndarray
    rhs:np.ndarray
        
        
class InteriorElementFactory(QuadratureTensor):
    def tri_multip(self,coeff:np.ndarray,degree:int)->np.ndarray:
        return np.dot(self.tri_quads[:degree,:degree,:degree],coeff) 
    def dub_multip(self,coeff:np.ndarray,degree:int)->np.ndarray:
        return np.dot(self.dub_quads[:degree,:degree],coeff)
    def generate_element(self,matfun:ChebyshevInterval,rhsfun:ChebyshevInterval):
        deg = matfun.degree
        h = matfun.h
        matel = self.tri_multip(matfun.coeffs,deg,)*h/2
        rhsel = self.dub_multip(rhsfun.coeffs,deg,)*h/2
        der_quads = self.der_dub_quads[:deg,:deg]
            
        return InteriorElement(der_quads,matel,rhsel)

'''

rule = u' - Au = 0 -> -v' - ATv = 0

'''
class AdjointInteriorElement(InteriorElement):
    def __init__(self,intel:InteriorElement):
        self.__dict__.update(intel.__dict__)        
    def to_matrix_form(self,):
        dim = self.rhs_element.shape[1]
        deg = self.mat_element.shape[1]        
        # takes the transpose for adjoint
        mat = self.mat_element.reshape([-1,deg,dim,dim]).transpose((0,2,1,3)).reshape([-1,deg*dim]) 
        rhs = self.rhs_element.flatten()
        der = self.der_quads.reshape([-1,1])*np.eye(dim).reshape([1,-1])
        der = der.reshape([-1,deg,dim,dim]).transpose((0,2,1,3)).reshape([-1,deg*dim])
        return InteriorElementMatrices(mat + der, rhs)
    
class AdjointInteriorElementFactory(InteriorElementFactory):
    def __init__(self,ief:InteriorElementFactory) -> None:
        self.__dict__.update(ief.__dict__)        
    def generate_element(self, matfun: ChebyshevInterval, rhsfun: ChebyshevInterval):
        return AdjointInteriorElement(super().generate_element(matfun, rhsfun))