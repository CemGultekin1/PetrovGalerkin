import logging
from .matalloc import BlockedMatrixFrame
import numpy as np
from scipy.sparse import lil_matrix
from .eqgen import LocalEquationFactory
from .matalloc import BlockedMatrixFrame
from chebyshev import GridwiseChebyshev

class GlobalSysAllocator:
    def __init__(self,dim:int,lcleq:LocalEquationFactory) -> None:
        self.local_equation = lcleq
        self.dim = dim
    def create_blocks(self,gridwise:GridwiseChebyshev):
        ps = gridwise.ps
        ncheb = len(ps)
        blocks = BlockedMatrixFrame(self.dim)
        if ncheb == 1:
            bclm0 = self.local_equation.generate_interior_matrices(gridwise.cheblist[0],1,1)
            bclm0.trim(left = True,right = True)
            blocks.add_tricolumn(bclm0)
        else:
            bclm0 = self.local_equation.generate_interior_matrices(gridwise.cheblist[0],1,ps[1])
            bclm0.trim(left = True)
            blocks.add_tricolumn(bclm0)
            for i,chebint in zip(range(1,ncheb-1),gridwise.cheblist[1:-1]):
                ldeg,rdeg = ps[i-1],ps[i+1]
                bclm = self.local_equation.generate_interior_matrices(chebint,ldeg,rdeg)
                blocks.add_tricolumn(bclm)
        
            bclm1 = self.local_equation.generate_interior_matrices(gridwise.cheblist[-1],ps[-2],1)
            bclm1.trim(right = True)
            blocks.add_tricolumn(bclm1)
        
        blocks.move(0,self.dim)
        blocks.extend(0,self.dim)
        
        bclm0 = self.local_equation.generate_boundary_blocks(1,1,ps[0])
        bclm0.trim(left = True,center = True)
        blocks.add_block(bclm0,)
        
        
        bclm1 = self.local_equation.generate_boundary_blocks(1,ps[-1],1)
        bclm1.trim(right = True,center = True)
        blocks.add_block(bclm1,upper_left=(-bclm1.height,-bclm1.width),)
        auxiliary_width = bclm1.width
        
        
        
        bc00,bc01 = self.local_equation.generate_boundary_condition_matrices(1,ps[-1])    
        bc10,bc11 = self.local_equation.generate_boundary_condition_matrices(ps[0],1)
        
          
        assert bc00.height == self.dim  and bc01.height == self.dim
        blocks.extend(bc00.height,0)  
        blocks.add_block(bc00,upper_left=(-bc00.height,0),rhs = True)
        blocks.add_block(bc01,upper_left=(-bc01.height,-bc01.width - auxiliary_width),rhs = False)
        
        
        
        assert bc10.height == self.dim  and bc11.height == self.dim
        blocks.extend(bc10.height,0)  
        blocks.add_block(bc10,upper_left=(-bc10.height,auxiliary_width),rhs = True)
        blocks.add_block(bc11,upper_left=(-bc11.height,-bc11.width),rhs = False)

        
        blocks.check_dimens()
        return blocks
    
class SparseGlobalSystem:
    mat:lil_matrix
    rhs:np.ndarray
    def __init__(self,blocks:BlockedMatrixFrame) -> None:
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
            
class DenseLocalSystem(SparseGlobalSystem):
    mat:np.ndarray
    def __init__(self,blocks:BlockedMatrixFrame) -> None:
        self.dim = blocks.dim
        mat = np.zeros(blocks.mat_shape)
        for blk in blocks.mat_blocks:
            logging.debug(f'blk.slicetpl = {blk.slicetpl},mat.shape = {mat.shape},blk.matblock.shape = {blk.matblock.shape}')
            mat[blk.slicetpl] += blk.matblock
        self.mat = mat
    
        
            
            
