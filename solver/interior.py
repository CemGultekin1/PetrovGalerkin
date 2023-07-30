from dataclasses import dataclass
import numpy as np
from chebyshev import ChebyshevInterval,QuadratureTensor
    

class InteriorElement:
    degree:int
    mat_element:np.ndarray # deg' x deg x (dim x dim)
    rhs_element:np.ndarray # deg' x dim
    der_quads:np.ndarray # deg' x deg
    def __init__(self,degree:int,der_quads:np.ndarray,mat_element:np.ndarray,rhs_element:np.ndarray) -> None:
        self.degree = degree
        self.der_quads = der_quads
        self.mat_element = mat_element        
        self.rhs_element = rhs_element    
        self.dim = self.rhs_element.shape[1]
    def mat_rhs_matrices(self,):
        deg = self.degree
        dim = self.dim
        mat = self.mat_element.reshape([-1,deg,dim,dim]).transpose((0,2,1,3)).reshape([-1,deg*dim]) 
        rhs = self.rhs_element.flatten()
        return mat,rhs
    def to_matrix_form(self,):
        mat,rhs = self.mat_rhs_matrices()
        deg = self.degree
        dim = self.dim
        der = self.der_quads.reshape([-1,1])*np.eye(dim).reshape([1,-1])
        der = der.reshape([-1,deg,dim,dim]).transpose((0,2,1,3)).reshape([-1,deg*dim])
        return InteriorElementMatrices(mat + der, rhs)


    
@dataclass
class InteriorElementMatrices:
    mat:np.ndarray
    rhs:np.ndarray
        
        
class InteriorElementFactory(QuadratureTensor):
    def tri_multip(self,coeff:np.ndarray,degree:int,multiplier :str= 'tri_quads')->np.ndarray:
        deg1 = coeff.shape[0]
        return np.dot(self.__dict__[multiplier][:degree,:degree,:deg1],coeff)     
    def dub_multip(self,coeff:np.ndarray,degree:int,multiplier :str= 'dub_quads')->np.ndarray:
        deg1 = coeff.shape[0]
        return np.dot(self.__dict__[multiplier][:degree,:deg1],coeff)    
    def generate_element(self,degree:int,matfun:ChebyshevInterval,rhsfun:ChebyshevInterval):
        deg = degree
        h = matfun.h
        dt = h/2
        matel = self.tri_multip(matfun.coeffs,deg,)*dt
        rhsel = self.dub_multip(rhsfun.coeffs,deg,)*dt
        der_quads = self.der_dub_quads[:deg,:deg]            
        return InteriorElement(degree,der_quads,matel,rhsel)
    def generate_quad_interior_time_derivatives_element(self,degree:int,matfun:ChebyshevInterval,rhsfun:ChebyshevInterval,left:bool = False):
        deg = degree
        h = matfun.h
        dt = h
        
        mat_one = self.tri_multip(matfun.coeffs,deg,multiplier='dtri_quads_one')/2
        mat_xhat = self.tri_multip(matfun.coeffs,deg,multiplier='dtri_quads_xhat')/2
        
        rhs_one = self.dub_multip(rhsfun.coeffs,deg,multiplier='ddub_quads_one')/2
        rhs_xhat = self.dub_multip(rhsfun.coeffs,deg,multiplier='ddub_quads_xhat')/2
        
        if left:
            matel = -1/dt * mat_one + 1/dt**2*mat_xhat
            rhsel = -1/dt * rhs_one + 1/dt**2*rhs_xhat
        else:
            matel = -1/dt * mat_one - 1/dt**2*mat_xhat
            rhsel = -1/dt * rhs_one - 1/dt**2*rhs_xhat
        return InteriorElement(degree,matel*0,matel,rhsel)
    def generate_quad_boundry_time_derivatives_element(self,deg:int,matfun:ChebyshevInterval,rhsfun:ChebyshevInterval,left:bool = False):
        if left:
            matel = self.tri_multip(matfun.coeffs,deg,multiplier='left_bndr_val_of_tri_quads')
            rhsel = self.dub_multip(rhsfun.coeffs,deg,multiplier='left_bndr_val_of_dub_quads')
        else:            
            matel = self.tri_multip(matfun.coeffs,deg,multiplier='right_bndr_val_of_tri_quads')
            rhsel = self.dub_multip(rhsfun.coeffs,deg,multiplier='right_bndr_val_of_dub_quads')
        return InteriorElement(deg,np.empty((0),),matel,rhsel)
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
        mat = self.mat_element.reshape([-1,deg,dim,dim]).transpose((0,3,1,2)).reshape([-1,deg*dim]) 
        rhs = self.rhs_element.flatten()
        der = self.der_quads.reshape([-1,1])*np.eye(dim).reshape([1,-1])
        der = der.reshape([-1,deg,dim,dim]).transpose((0,2,1,3)).reshape([-1,deg*dim])
        return InteriorElementMatrices(mat + der, rhs)
    
class AdjointInteriorElementFactory(InteriorElementFactory):
    def __init__(self,ief:InteriorElementFactory) -> None:
        self.__dict__.update(ief.__dict__)        
    def generate_element(self,*args):
        return AdjointInteriorElement(super().generate_element(*args))