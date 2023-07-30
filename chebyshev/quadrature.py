from typing import Tuple
from .element  import Degree
import numpy.polynomial.chebyshev as cheb
import numpy as np


def quadfun_bndrs(*degrees:int, derinds:Tuple[int,...] = ()):
    y = np.ones([1],dtype = float)
    for i,deg in enumerate(degrees):
        x = np.zeros((deg+1),dtype = float)
        x[deg] = 1
        if i in derinds:
            x = cheb.chebder(x)
        y = cheb.chebmul(y,x)
    return - cheb.chebval(-1,y),cheb.chebval(1,y)

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
    
    
def dquadfun(*degrees:int, derinds:Tuple[int,...] = (),multiply:Tuple[int,...] = ()):
    y = np.ones([1],dtype = float)
    for i,deg in enumerate(degrees):
        x = np.zeros((deg+1),dtype = float)
        x[deg] = 1
        if i in derinds:
            x = cheb.chebder(x)
            if i in multiply:
                z = np.zeros((2),dtype = float)
                z[1] = 1
                x = cheb.chebmul(x,z)
        y = cheb.chebmul(y,x)
    yint = cheb.chebint(y)
    return cheb.chebval(1,yint) - cheb.chebval(-1,yint)

def ddquadfun(*degrees:int, derinds:Tuple[int,...] = (),multiply:Tuple[int,...] = ()):
    y = np.ones([1],dtype = float)
    for i,deg in enumerate(degrees):
        x = np.zeros((deg+1),dtype = float)
        x[deg] = 1
        if i in derinds:
            x = cheb.chebder(x)
            x = cheb.chebder(x)
            if i in multiply:
                z = np.zeros((2),dtype = float)
                z[1] = 1
                x = cheb.chebmul(x,z)
        y = cheb.chebmul(y,x)
    yint = cheb.chebint(y)
    return cheb.chebval(1,yint) - cheb.chebval(-1,yint)


class InnerProducts(Degree):
    def __init__(self,degree:int) -> None:
        super().__init__(degree)
        self.dub_quads = np.empty((self.degree,self.degree,),dtype = float)
        self.ddub_quads_one = np.empty((self.degree,self.degree,),dtype = float)
        self.ddub_quads_xhat = np.empty((self.degree,self.degree,),dtype = float)
        self.right_bndr_val_of_dub_quads = np.empty((self.degree,self.degree,),dtype = float)
        self.left_bndr_val_of_dub_quads = np.empty((self.degree,self.degree,),dtype = float)
        
        # dtri_quads_right_one_dt
        # dtri_quads_right_two_dt
        # dtri_quads_left_one_dt
        # dtri_quads_left_two_dt
        # left_bndr_val_of_tri_quads
        # right_tbndr_val_of_ri_quads
        self.filledup = False
    def fillup(self,):
        '''
        
        xhat = (x - (t2+t1)/2)/(t2 - t1)*2
        xhat(t1) = -1
        xhat(t2) = +1
        
        d xhat /d t1 = - 1/dt + xhat /dt = -(1 - xhat)/dt
        d xhat /d t2 = - 1/dt - xhat /dt = -(1 + xhat)/dt
        '''
        for i,j in self.degree_index_product(2):
            self.dub_quads[i,j] = quadfun(i,j)
            self.ddub_quads_xhat[i,j] = dquadfun(i,j,derinds=(0,),multiply=(0,)) 
            self.ddub_quads_one[i,j] = dquadfun(i,j,derinds=(0,),multiply=())  
            
                        
            left,right = quadfun_bndrs(i,j,)
            self.right_bndr_val_of_dub_quads[i,j] = right
            self.left_bndr_val_of_dub_quads[i,j] =  left
        self.filledup = True
        
class QuadratureTensor(InnerProducts):
    def __init__(self,degree:int) -> None:
        super().__init__(degree)
        tri = (self.degree,self.degree,self.degree)
        dub = (self.degree,self.degree)
        self.tri_quads = np.empty(tri,dtype = float)
        self.dtri_quads_one = np.empty(tri,dtype = float)
        self.dtri_quads_xhat= np.empty(tri,dtype = float)
        
        self.left_bndr_val_of_tri_quads = np.empty(tri,dtype = float)
        self.right_bndr_val_of_tri_quads = np.empty(tri,dtype = float)
         
        self.der_dub_quads = np.empty(dub,dtype = float)
       
        self.filledup = False
    def fillup(self,):
        '''
        xhat = (x - (t2 + t1)/2)/(t2 - t1)*2
        d xhat /d t1 = - 1/dt + xhat /dt = -(1 - xhat)/dt
        d xhat /d t2 = - 1/dt - xhat /dt = -(1 + xhat)/dt
        '''
        super().fillup()
        for i,j,k in self.degree_index_product(3):
            self.tri_quads[i,j,k] = -quadfun(i,j,k)
            
            self.dtri_quads_one[i,j,k] = -dquadfun(i,j,k,derinds = (0,),multiply = ()) \
                                        -dquadfun(i,j,k,derinds = (1,),multiply = ())
            self.dtri_quads_xhat[i,j,k] = -dquadfun(i,j,k,derinds = (0,),multiply = (0,)) \
                                        -dquadfun(i,j,k,derinds = (1,),multiply = (1,))

            left,right = quadfun_bndrs(i,j,k,derinds=()) # _/ v*u*A  on the boundaries
            self.left_bndr_val_of_tri_quads[i,j,k] = -left
            self.right_bndr_val_of_tri_quads[i,j,k] = -right
            
        for i,j in self.degree_index_product(2):
            self.der_dub_quads[i,j] = -quadfun(i,j,first_derivative=True)
       
        self.filledup = True