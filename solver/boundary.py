from dataclasses import dataclass
import numpy.polynomial.chebyshev as cheb
import numpy as np
from .bndrcond import BoundaryCondition
from chebyshev import Value,Boundary,degree_mat_multip
                      
            
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
        dub = (self.degree,self.degree)
        self.center = np.empty(dub,dtype = float)  
        self.right_cross = np.empty(dub,dtype = float)
        self.left_cross = np.empty(dub,dtype = float)
        
        self.dcenter_on_right = np.empty(dub,dtype = float)  
        self.dcenter_on_left = np.empty(dub,dtype = float)  


        self.dright_cross_dright = np.empty(dub,dtype = float)
        self.dright_cross_dcenter = np.empty(dub,dtype = float)
        
        self.dleft_cross_dcenter = np.empty(dub,dtype = float)        
        self.dleft_cross_dleft = np.empty(dub,dtype = float)        
        
        self.value = Value(degree)
        self.filledup = False
    def fillup(self,):
        self.value.fillup()
        for i,j in self.degree_index_product(2):
            self.center[i,j] =  self.value.right[i]*self.value.right[j]/2 - self.value.left[i]*self.value.left[j]/2
            self.right_cross[i,j] =  -self.value.left[i]*self.value.right[j]/2
            self.left_cross[i,j] =  self.value.right[i]*self.value.left[j]/2
            
            self.dcenter_on_right[i,j] =  self.value.dright[i]*self.value.right[j]/2 + self.value.right[i]*self.value.dright[j]/2
            self.dcenter_on_left[i,j] = - self.value.dleft[i]*self.value.left[j]/2  - self.value.left[i]*self.value.dleft[j]/2
            
            self.dright_cross_dright[i,j] =  -self.value.dleft[i]*self.value.right[j]/2
            self.dright_cross_dcenter[i,j] =  -self.value.left[i]*self.value.dright[j]/2
            
            self.dleft_cross_dcenter[i,j] =  self.value.right[i]*self.value.dleft[j]/2 
            self.dleft_cross_dleft[i,j] =  self.value.dright[i]*self.value.left[j]/2
        self.filledup = True
    def generate_element(self,degree1:int,degree2:int,degree3:int,):
        bcross = self.left_cross[:degree1,:degree2]
        fcross = self.right_cross[:degree3,:degree2]
        base = self.center[:degree2,:degree2]
        return BoundaryElement(bcross,base,fcross)
    def generate_time_derivative_element(self,degree1:int,degree2:int,degree3:int,\
                                            h1:float,h2:float,h3:float,\
                                            left: bool = True):
        if left:
            bcross = self.dleft_cross_dcenter[:degree1,:degree2]/h2*2 \
                        + self.dleft_cross_dleft[:degree1,:degree2]/h1*2
            fcross = self.dright_cross_dcenter[:degree3,:degree2]*0
            base = self.dcenter_on_left[:degree2,:degree2]/h2*2
        else:
            bcross = self.dleft_cross_dcenter[:degree1,:degree2]*0        
            fcross = self.dright_cross_dcenter[:degree3,:degree2]/h2*2 \
                        + self.dright_cross_dright[:degree3,:degree2]/h3*2
            base = self.dcenter_on_right[:degree2,:degree2]/h2*2
        return BoundaryElement(bcross,base,fcross)
    def averaging_edge_value(self,leftdegree:int,rightdegree:int):        
        leftval = self.value.right[:leftdegree]/2
        rightval = self.value.left[:rightdegree]/2
        return leftval,rightval
    def averaging_time_derivative_edge_value(self,leftdegree:int,rightdegree:int):        
        leftval = self.value.dright[:leftdegree]/2
        rightval = self.value.dleft[:rightdegree]/2
        return leftval,rightval
    def create_boundary_condition_element_factory(self,bc:BoundaryCondition,):
        return BoundaryConditionElementFactory(bc,self.value)
def eye_kron_multip(vec:np.ndarray,eyevec:np.ndarray):
    if vec.size == 0:
        return np.empty((0,eyevec.shape[1]*vec.shape[1]))
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
class BaseBoundaryElement(BoundaryElement):
    base_element:np.ndarray
    def __init__(self,base_element:np.ndarray,) -> None:
        d = base_element.shape[1]
        super().__init__(np.empty((0,d)),base_element,np.empty((0,d)))
        
    
@dataclass
class BoundaryElementMatrices:
    mat_left:np.ndarray
    mat_center:np.ndarray
    mat_right:np.ndarray
    

@dataclass
class BoundaryConditionElementMatrices:
    mat_b0:np.ndarray
    mat_b1:np.ndarray
    rhs_c:np.ndarray

class BoundaryConditionElement:
    def __init__(self,left_most_left:np.ndarray,right_most_right:np.ndarray,\
                        b0:np.ndarray,b1:np.ndarray,c:np.ndarray) -> None:
            self.mat_left_most_left = left_most_left
            self.mat_right_most_right = right_most_right
            self.b0,self.b1,self.c = b0,b1,c
    def to_matrix_form(self,)->'BoundaryConditionElementMatrices':
        b0_interior = degree_mat_multip(self.mat_left_most_left,self.b0)
        b1_interior = degree_mat_multip(self.mat_right_most_right,self.b1)
        return BoundaryConditionElementMatrices(b0_interior,b1_interior,self.c)