from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple


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
        self._matclm :List[np.ndarray]=[mat_left,mat_center,mat_right]
        self.rhsclm = rhs_center
        self.width = self._matclm[1].shape[1]
    @property
    def sides(self,):
        return tuple(mat.shape[0] for mat in self._matclm)
    @property
    def matclm(self,):
        return np.concatenate(self._matclm,axis = 0)
    def trim(self,left:bool = False,right:bool = False):
        if left:
            self._matclm[0] = np.empty((0,self._matclm[0].shape[1]))
        if right:
            self._matclm[-1] = np.empty((0,self._matclm[-1].shape[1]))
            
class BlockColumnsBndrCorrection:
    def __init__(self,bdrmat:BoundaryElementMatrices,**leftright:bool) -> None:
        self.matclm = bdrmat.mat_center
        self.leftright = leftright
    def is_left(self,)->bool:
        if 'left'in self.leftright:
            return self.leftright['left']
        elif 'right' in self.leftright:
            return not self.leftright['right']
    def is_right(self,)->bool:
        return not self.is_left()
    def correct_block_column(self,block_clm:BlockColumns,):
        block_clm._matclm[1] += self.matclm
        block_clm.trim(**self.leftright)
        return block_clm
        
            
            

class LocalEquationFactory:
    def __init__(self,dim:int,max_degree:int,boundary_condition:BoundaryCondition) -> None:        
        self.dim = dim
        self.interr = InteriorElementFactory(max_degree)
        self.interr.fillup()
        self.bndr = BoundaryElementFactory(max_degree)        
        self.bndr.fillup()
        self.bndr_cond = self.bndr.create_boundary_condition_element_factory(boundary_condition)
        
    def generate_interior_matrices(self,funs:FlatListOfFuns,\
                    center_chebint:ChebyshevInterval,\
                            left_degree:int,right_degree:int,\
                            minus_one_test_degree:bool = False):
        center_degree = center_chebint.degree
        matfun,rhsfun = center_chebint.separate_funs(funs)
        interrelem = self.interr.generate_element(matfun,rhsfun,minus_one_test_degree=minus_one_test_degree).to_matrix_form()
        bndrelem = self.bndr.generate_element(left_degree,center_degree,right_degree,\
            minus_one_test_degree=minus_one_test_degree).to_matrix_form(self.dim)
        return BlockColumns(interrelem,bndrelem)
    
    def generate_edge_boundary_correction_matrices(self,center_degree:int,minus_one_test_degree:bool=False,**leftright:bool):
        bndrelem = self.bndr.generate_edge_element_correction(center_degree,\
                    minus_one_test_degree = minus_one_test_degree,**leftright).to_matrix_form(self.dim)
        return BlockColumnsBndrCorrection(bndrelem,**leftright)   
            
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
    
def tri_column_slices(a:int,b:int,c:int,p:int,w,p1:int):
    return slice(p-a,p+b+c),slice(p1,p1+w)
def rhs_column_slices(b:int,p:int):
    return slice(p,p+b)

class ColRowCounter(list):
    def __init__(self,col:int,row:int) -> None:
        list.__init__(self,)
        self.append(col)
        self.append(row)
    def add(self,col,row):
        self[0] += col
        self[1] += row
    def __str__(self,):
        return f'({self[0]},{self[1]})'
class BlockColumnsList:
    def __init__(self,dim:int):
        self.dim = dim
        self.mat_blocks :List[AllocatableBlock]= []
        self.rhs_blocks :List[AllocatableBlock]= []
        self.crc = ColRowCounter(self.dim,0)
        
    def add_interior(self,mc:BlockColumns):
        a,b,c = mc.sides
        w = mc.width
        pcol,prow = self.crc
        matslctp = tri_column_slices(a,b,c,pcol,w,prow)
        rhsslctp = rhs_column_slices(b,pcol)
        ab = AllocatableBlock(mc.matclm,matslctp)
        rhsab = AllocatableBlock(mc.rhsclm,rhsslctp)
        self.mat_blocks.append(ab)
        self.rhs_blocks.append(rhsab)
        self.crc.add(b,w)
        logging.debug(f'Interior block added with rows: {(a,b,c)}, self.crc = {self.crc}')
        # logging.debug(f'\t\t\t slice tuple = {matslctp}')
        # return ab,rhsab 
    # def decrement_prncpl(self,):
    #     self.prncpl -= self.dim
    def add_boundary(self,bcem:BoundaryConditionElementMatrices):
        p = bcem.mat_b0.shape[0]
        lside = bcem.mat_b0.shape[1]
        rside = bcem.mat_b1.shape[1]

        tplinnerleft = (slice(0,p),slice(0,lside))
        _,prow = self.crc
        tplinnerright = (slice(0,p),slice(prow - rside,prow))
        
        logging.debug(f'Boundary blocks added with tuple slices: {tplinnerleft}, and {tplinnerright}')
        
        abrhs = AllocatableBlock(bcem.rhs_c,(slice(0,p)))

        abinnerleft = AllocatableBlock(bcem.mat_b0,tplinnerleft)
        abinnerright = AllocatableBlock(bcem.mat_b1,tplinnerright)
        self.mat_blocks.extend((abinnerleft,abinnerright))
        self.rhs_blocks.append(abrhs)
    @property
    def mat_shape(self,):
        return tuple(self.crc)

    @property
    def rhs_shape(self,):
        return (self.crc[0],)
    def check_dimens(self,):
        shp = self.mat_shape
        for i,mat in enumerate(self.mat_blocks):
            if not mat.is_inside(*shp):
                logging.error(f'Trimatrix column #{i} of shape {mat.shape()} \n\t and slice {mat.slicetpl} doesn\'t sit within the global shape = {shp}')
                raise Exception
        
        
        
class GlobalSysAllocator:
    def __init__(self,dim:int,funs:FlatListOfFuns,lcleq:LocalEquationFactory) -> None:
        self.local_equation = lcleq
        self.dim = dim
        self.flof = funs
    def create_blocks(self,gridwise:GridwiseChebyshev):
        ps = gridwise.ps
        ncheb = len(ps)
        blocks = BlockColumnsList(self.dim)
        left_edge_block_corr = self.local_equation.generate_edge_boundary_correction_matrices(ps[0],left= True,minus_one_test_degree=True)
        
        if ncheb == 1:
            right_edge_block_corr = self.local_equation.generate_edge_boundary_correction_matrices(ps[-1],right = True,minus_one_test_degree=True)
            bclm = self.local_equation.generate_interior_matrices(self.flof, gridwise.cheblist[0],1,1,minus_one_test_degree=True)
            bclm = left_edge_block_corr.correct_block_column(bclm)
            bclm = right_edge_block_corr.correct_block_column(bclm)
            logging.debug(f'block matrix shape = {bclm.matclm.shape}')
            blocks.add_interior(bclm)
        else:
            right_edge_block_corr = self.local_equation.generate_edge_boundary_correction_matrices(ps[-1],right = True)
            bclm0 = self.local_equation.generate_interior_matrices(self.flof, gridwise.cheblist[0],1,ps[1],minus_one_test_degree=True)            
            bclm0 = left_edge_block_corr.correct_block_column(bclm0)            
            blocks.add_interior(bclm0)
            
            for i,chebint in zip(range(1,ncheb-1),gridwise.cheblist[1:-1]):
                ldeg,rdeg = ps[i-1],ps[i+1]
                bclm = self.local_equation.generate_interior_matrices(self.flof, chebint,ldeg,rdeg)
                blocks.add_interior(bclm)
                
            bclm1 = self.local_equation.generate_interior_matrices(self.flof, gridwise.cheblist[-1],ps[-2],1)
            bclm1 = right_edge_block_corr.correct_block_column(bclm1)
            blocks.add_interior(bclm1)
            
        bcem = self.local_equation.generate_boundary_condition_matrices(ps[0],ps[-1])
        blocks.add_boundary(bcem)
        blocks.check_dimens()
        return blocks

class SparseGlobalSystem:
    mat:lil_matrix
    rhs:np.ndarray
    def __init__(self,blocks:BlockColumnsList) -> None:
        self.dim = blocks.dim
        mat = lil_matrix(blocks.mat_shape)
        rhs = np.zeros(blocks.rhs_shape)
        for blk in blocks.mat_blocks:
            logging.debug(f'blk.slicetpl = {blk.slicetpl},mat.shape = {mat.shape},blk.matblock.shape = {blk.matblock.shape}')
            mat[blk.slicetpl] += blk.matblock
        for blk in blocks.rhs_blocks:
            rhs[blk.slicetpl] += blk.matblock
        self.mat = mat.tocsr()
        self.rhs = rhs
            
        



