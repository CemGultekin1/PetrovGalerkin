from dataclasses import dataclass
import logging
from typing import List, Tuple


from chebyshev.core import BoundaryCondition
from .boundary import BoundaryElementMatrices,BoundaryElementFactory,BoundaryConditionElementMatrices
from .interior import InteriorElementMatrices,InteriorElementFactory
from .interval import GridwiseChebyshev,ChebyshevInterval
from .funs import FlatListOfFuns
from scipy.sparse import lil_matrix
import numpy as np
class BlockColumns:
    def __init__(self,interrmat:InteriorElementMatrices,bdrmat:BoundaryElementMatrices) -> None:
        mat_center = interrmat.mat + bdrmat.mat_center
        rhs_center = interrmat.rhs
        mat_left = bdrmat.mat_left
        mat_right = bdrmat.mat_right
        self.matclm = np.concatenate([mat_left,mat_center,mat_right],axis = 0)
        self.rhsclm = rhs_center
        self.sides = (mat_left.shape[0],mat_center.shape[0],mat_right.shape[0])
    def trim_last_row(self,):
        a,b,_ = self.sides
        self.matclm = self.matclm[:a+b,:]
        self.sides = a,b,0
        
        
class LocalEquationFactory:
    def __init__(self,dim:int,max_degree:int,boundary_condition:BoundaryCondition) -> None:        
        self.dim = dim
        self.interr = InteriorElementFactory(max_degree)
        self.interr.fillup()
        self.bndr = BoundaryElementFactory(max_degree)        
        self.bndr.fillup()
        self.bndr_cond = self.bndr.create_boundary_condition_element_factory(boundary_condition)
        
    def generate_interior_matrices(self,funs:FlatListOfFuns,center_chebint:ChebyshevInterval,left_degree:int,right_degree:int):
        center_degree = center_chebint.degree
        matfun,rhsfun = center_chebint.separate_funs(funs)
        interrelem = self.interr.generate_element(matfun,rhsfun).to_matrix_form()
        bndrelem = self.bndr.generate_element(left_degree,center_degree,right_degree).to_matrix_form(self.dim)
        return BlockColumns(interrelem,bndrelem)
    def generate_boundary_condition_matrices(self,leftmost_degree:int,rightmost_degree:int):
        bce = self.bndr_cond.generate_element(leftmost_degree,rightmost_degree)
        return bce.to_matrix_form()
@dataclass
class AllocatableBlock:
    matblock:np.ndarray
    slicetpl:Tuple[slice,...]
    def is_inside(self,*shp:int):
        for slci,shpi in zip(self.slicetpl,shp):
            if slci.stop > shpi:
                return False
        return True
    def shape(self,):
        return tuple(slc.stop  - slc.start for slc in self.slicetpl)
def tri_column_slices(a:int,b:int,c:int,p:int):
    return slice(p-a,p+b+c),slice(p,p+b)
def rhs_column_slices(b:int,p:int):
    return slice(p,p+b)

class BlockColumnsList:
    def __init__(self,dim:int):
        self.mat_blocks :List[AllocatableBlock]= []
        self.rhs_blocks :List[AllocatableBlock]= []
        self.prncpl = dim
    def add_interior(self,mc:BlockColumns):
        a,b,c = mc.sides
        p = self.prncpl 
        matslctp = tri_column_slices(a,b,c,p)
        rhsslctp = rhs_column_slices(b,p)
        ab = AllocatableBlock(mc.matclm,matslctp)
        rhsab = AllocatableBlock(mc.rhsclm,rhsslctp)
        self.mat_blocks.append(ab)
        self.rhs_blocks.append(rhsab)
        self.prncpl += b
    def add_boundary(self,bcem:BoundaryConditionElementMatrices):
        p = bcem.mat_b0.shape[0]
        lside = bcem.mat_b0.shape[1]
        rside = bcem.mat_b1.shape[1]
        tplouter = (slice(0,p),slice(0,p))
        tplinnerleft = (slice(0,p),slice(p,p + lside))
        tplinnerright = (slice(0,p),slice(self.prncpl - rside,self.prncpl))
        abrhs = AllocatableBlock(bcem.rhs_c,(slice(0,p)))
        abouter = AllocatableBlock(bcem.mat_outer,tplouter)
        abinnerleft = AllocatableBlock(bcem.mat_b0,tplinnerleft)
        abinnerright = AllocatableBlock(bcem.mat_b1,tplinnerright)
        self.mat_blocks.extend((abouter,abinnerleft,abinnerright))
        self.rhs_blocks.append(abrhs)
    @property
    def mat_shape(self,):
        return (self.prncpl,self.prncpl)

    @property
    def rhs_shape(self,):
        return (self.prncpl,)
    def check_dimens(self,):
        shp = self.mat_shape
        for i,mat in enumerate(self.mat_blocks):
            if not mat.is_inside(*shp):
                logging.error(f'Trimatrix column #{i} of shape \
                        {mat.shape()} \n\t and slice {mat.slicetpl} doesn\'t \
                            sit within the global shape = {shp}')
                raise Exception
        
        
        
class NonAllocatedGlobalSystem:
    def __init__(self,dim:int,funs:FlatListOfFuns,lcleq:LocalEquationFactory) -> None:
        self.local_equation = lcleq
        self.dim = dim
        self.flof = funs
    def create_blocks(self,gridwise:GridwiseChebyshev):
        blocks = BlockColumnsList(self.dim)
        degrees = [1] + gridwise.ps + [1]
        np2 = len(degrees)
        for i,chebint in zip(range(1,np2-1),gridwise.cheblist):
            ldeg,rdeg = degrees[i-1],degrees[i+1]
            bclm = self.local_equation.generate_interior_matrices(self.flof, chebint,ldeg,rdeg)
            if i == np2 -2:
                bclm.trim_last_row()
            blocks.add_interior(bclm)
        bcem = self.local_equation.generate_boundary_condition_matrices(degrees[1],degrees[-2])
        blocks.add_boundary(bcem)
        blocks.check_dimens()
        return blocks

class SparseGlobalSystem:
    mat:lil_matrix
    rhs:np.ndarray
    def __init__(self,blocks:BlockColumnsList) -> None:
        mat = lil_matrix(blocks.mat_shape)
        rhs = np.zeros(blocks.rhs_shape)
        for blk in blocks.mat_blocks:
            logging.debug(f'blk.slicetpl = {blk.slicetpl},mat.shape = {mat.shape},blk.matblock.shape = {blk.matblock.shape}')
            mat[blk.slicetpl] = blk.matblock
        for blk in blocks.rhs_blocks:
            rhs[blk.slicetpl] = blk.matblock
        self.mat = mat
        self.rhs = rhs
            
        



