import numpy.polynomial.chebyshev as cheb
import numpy as np
from .element import Degree
class Boundary(Degree):
    def __init__(self,degree:int) -> None:
        super().__init__(degree)
        self.degree = degree
        self.front = np.empty((self.degree,),dtype = float)
        self.back = np.empty((self.degree,),dtype = float)
    def fillup(self,):...
    
def flux_element(deg:int,front:bool = True):
    x = np.zeros((deg+1),dtype = float)
    x[deg] = 1
    if front:
        return -cheb.chebval(1,x)
    else:
        return cheb.chebval(-1,x)

def mean_element(deg:int,front:bool = True):
    x = np.zeros((deg+1),dtype = float)
    x[deg] = 1
    if front:
        return cheb.chebval(1,x)/2
    else:
        return cheb.chebval(-1,x)/2
    
class Flux(Boundary):
    def fillup(self,):
        for i in self.degree_index_product(1):
            self.front[i] =  flux_element(i,front=True)
            self.back[i] =  flux_element(i,front=False)
            
class Mean(Boundary):
    def fillup(self,):
        for i in self.degree_index_product(1):
            self.front[i] =  mean_element(i,front=True)
            self.back[i] =  mean_element(i,front=False)

class Edge(Mean):
    def fillup(self):
        super().fillup()
        self.front *=2
        self.back *=2
    

class QuadraticBoundary(Boundary):
    def __init__(self,degree:int) -> None:
        self.degree = degree
        self.base_element = np.empty((self.degree,self.degree),dtype = float)        
        self.front_cross = np.empty((self.degree,self.degree),dtype = float)
        self.back_cross = np.empty((self.degree,self.degree),dtype = float)
        self.flux = Flux()
        self.mean = Mean()
    def fillup(self,):
        self.flux.fillup()
        self.mean.fillup()
        for i,j in self.degree_index_product(2):
            self.base_element[i,j] =  self.flux.front[i]*self.mean.front[j] + self.flux.back[i]*self.mean.front[j]
            
            self.front_cross[i,j] =  self.flux.front[i]*self.mean.back[j] 
            self.back_cross[i,j] =  self.flux.back[i]*self.mean.front[j]
    def create_boundary_element(self,degree1:int,degree2:int,degree3:int):
        bcross = self.back_cross[:degree1,:degree2]
        fcross = self.front_cross[:degree2,:degree3]
        base = self.base_element[:degree2,:degree2]
        return BoundaryElement(bcross,base,fcross)
    
class BoundaryElement:
    front_cross_element:np.ndarray 
    base_element:np.ndarray
    back_cross_element:np.ndarray     
    def __init__(self,back_cross_element:np.ndarray ,base_element:np.ndarray,front_cross_element:np.ndarray) -> None:
        self.front_cross_element = front_cross_element
        self.base_element = base_element
        self.back_cross_element = back_cross_element
    
    
# class GlobalBoundaryEquation(LinearElement):
#     left_boundary_element:np.ndarray
#     right_boundary_element:np.ndarray
#     def __init__(self,left_boundary_element:np.ndarray,right_boundary_element:np.ndarray) -> None:
#         super().__init__()
#         self.left_boundary_element = left_boundary_element
#         self.right_boundary_element = right_boundary_element    

# class BoundaryElementFactory(QuadraticBoundary):
#     def base_boundary_elements(self,):
#         return self.base_element
#     def forward_cross_boundary_elements(self,):
#         return self.front_cross
#     def backward_cross_boundary_elements(self,):
#         return self.back_cross
#     def create_elements(self,):
#         be = self.base_boundary_elements()
#         fce = self.forward_cross_boundary_elements()
#         bce = self.backward_cross_boundary_elements()
#         return BoundaryElement(self.dim,self.degree,fce,be,bce)
        