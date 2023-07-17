from dataclasses import dataclass
import numpy.polynomial.chebyshev as cheb
import numpy as np

from chebyshev.core import BoundaryCondition
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


def degree_mat_multip(degmat:np.ndarray,bdrmat:np.ndarray):
    
    bdrdeg = degmat.reshape([1,-1,1])*np.stack([bdrmat],axis = 1)
    return bdrdeg
                      
            
class BoundaryConditionElementFactory:
    def __init__(self,bc:BoundaryCondition,boundary_values:Value,outside_degree:int = 1) -> None:
        self.boundary_condition = bc
        self.boundary_values = boundary_values
        self.outside_degree = outside_degree
        
    def generate_element(self,left_most_degree:int,right_most_degree:int):
        
        left_most_left = self.boundary_values.left[:left_most_degree]
        right_most_right = self.boundary_values.right[:right_most_degree]
        return BoundaryConditionElement(left_most_left,right_most_right,\
            self.boundary_condition.B0,self.boundary_condition.B1,self.boundary_condition.c)
        
class BoundaryElementFactory(Boundary):
    def __init__(self,degree:int) -> None:
        self.degree = degree
        self.base_element = np.empty((self.degree,self.degree),dtype = float)        
        self.right_cross = np.empty((self.degree,self.degree),dtype = float)
        self.left_cross = np.empty((self.degree,self.degree),dtype = float)
        self.flux = Flux(degree)
        self.value = Value(degree)
        self.filledup = False
    def fillup(self,):
        self.flux.fillup()
        self.value.fillup()
        for i,j in self.degree_index_product(2):
            self.base_element[i,j] =  self.flux.right[i]*self.value.right[j] + self.flux.left[i]*self.value.right[j]
            
            self.right_cross[i,j] =  self.flux.right[i]*self.value.left[j] 
            self.left_cross[i,j] =  self.flux.left[i]*self.value.right[j]
        self.filledup = True
    def generate_element(self,degree1:int,degree2:int,degree3:int):
        bcross = self.left_cross[:degree1,:degree2]
        fcross = self.right_cross[:degree3,:degree2]
        base = self.base_element[:degree2,:degree2]
        return BoundaryElement(bcross,base,fcross)
    def create_boundary_condition_element_factory(self,bc:BoundaryCondition,):
        return BoundaryConditionElementFactory(bc,self.value)
def eye_kron_multip(vec:np.ndarray,eyevec:np.ndarray):
    vec1 =  vec.reshape([vec.shape[0],1,vec.shape[1],1])*eyevec
    return vec1.reshape([vec.shape[0]*eyevec.shape[1],-1])
class BoundaryElement:
    left_cross_element:np.ndarray  # rd x cd
    base_element:np.ndarray # cd x cd
    right_cross_element:np.ndarray  # ld x cd
    def __init__(self,left_cross_element:np.ndarray ,base_element:np.ndarray,right_cross_element:np.ndarray) -> None:
        self.right_cross_element = right_cross_element
        self.base_element = base_element
        self.left_cross_element = left_cross_element
    def to_matrix_form(self,dim:int)->'BoundaryElementMatrices':
        eye = np.eye(dim).reshape([1,dim,1,dim])
        lmat,cmat,rmat = (eye_kron_multip(vec,eye) for vec in (self.left_cross_element,self.base_element,self.right_cross_element))
        return BoundaryElementMatrices(lmat,cmat,rmat)
@dataclass
class BoundaryElementMatrices:
    mat_left:np.ndarray
    mat_center:np.ndarray
    mat_right:np.ndarray
    

@dataclass
class BoundaryConditionElementMatrices:
    mat_outer:np.ndarray
    mat_b0:np.ndarray
    mat_b1:np.ndarray
    rhs_c:np.ndarray

class BoundaryConditionElement:
    def __init__(self,left_most_left:np.ndarray,right_most_right:np.ndarray,\
                        b0:np.ndarray,b1:np.ndarray,c:np.ndarray) -> None:
            outer = b0 + b1
            self.mat_outer = outer
            self.mat_left_most_left = left_most_left
            self.mat_right_most_right = right_most_right
            self.b0,self.b1,self.c = b0,b1,c
            self.dim = b0.shape[0]
    def to_matrix_form(self,)->'BoundaryConditionElementMatrices':
        b0_interior = degree_mat_multip(self.mat_left_most_left/2,self.b0).reshape([self.dim,-1])      
        b1_interior = degree_mat_multip(self.mat_right_most_right/2,self.b1).reshape([self.dim, -1])
        return BoundaryConditionElementMatrices(self.mat_outer,b0_interior,b1_interior,self.c)