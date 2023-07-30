from dataclasses import dataclass
import logging
from typing import  List, Tuple
from .boundary import BoundaryElementMatrices
from .interior import InteriorElementMatrices
import numpy as np

class AllocationSystem:
    def __init__(self,ps:np.ndarray,dim:int) -> None:
        ps = np.insert(ps,0,1)
        ps = np.insert(ps,len(ps),1)
        self.degrees = ps
        self.dim = dim
        self.right_edges = np.cumsum(ps)*dim
    def get_interval_center(self,i:int)->Tuple[int,int]:
        i+=1
        redge = self.right_edges[i]
        ledge = 0
        if i > 0:
            legde = self.right_edges[i-1]
        return ledge,redge
    
        
class TriRowColumn:
    def __init__(self,mat_left:np.ndarray,mat_center:np.ndarray,mat_right:np.ndarray,rhs_center:np.ndarray) -> None:
        mats = (mat_left,mat_center,mat_right)
        not_none = [mat is not None for mat in mats]
        if sum(not_none) == 0:
            raise Exception
        notnonei = not_none.index(True)
        self.width = mats[notnonei].shape[1]
        self._matclm :List[np.ndarray] = [mat if notnone else np.empty((0,self.width)) for mat,notnone in zip(mats,not_none)]
        self.rhsclm = rhs_center
        
    @property
    def sides(self,):
        return tuple(mat.shape[0] for mat in self._matclm)
    @property
    def matclm(self,):
        return np.concatenate(self._matclm,axis = 0)
    @property
    def height(self,):
        return sum(self.sides)
    def add(self,trc:'TriRowColumn'):
        for i,(lmat,rmat) in enumerate(zip(self._matclm,trc._matclm)):
            self._matclm[i] = lmat+rmat
        self.rhsclm += trc.rhsclm
            
    def trim(self,left:bool = False,right:bool = False,center:bool = False):
        if left:
            self._matclm[0] = np.empty((0,self.width))
        if right:
            self._matclm[-1] = np.empty((0,self.width))
        if center:
            self._matclm[1] = np.empty((0,self.width))

class BlockColumns(TriRowColumn):
    def __init__(self,interrmat:InteriorElementMatrices,bdrmat:BoundaryElementMatrices) -> None:
        mat_center = interrmat.mat + bdrmat.mat_center
        rhs_center = interrmat.rhs
        mat_left = bdrmat.mat_left
        mat_right = bdrmat.mat_right
        super().__init__(mat_left,mat_center,mat_right,rhs_center)
            


   
def tri_column_slices(a:int,b:int,c:int,p:int,w,p1:int):
    return slice(p-a,p+b+c),slice(p1,p1+w)
def rhs_column_slices(b:int,p:int):
    return (slice(p,p+b),)

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
    def move(self,*stps:int):
        slicelist = list(self.slicetpl)
        for i,(slci,stpi) in enumerate(zip(slicelist,stps)):
            slicelist[i] = slice(slci.start + stpi,slci.stop + stpi)
        self.slicetpl = tuple(slicelist)
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
    def move(self,col,row):
        self.add(col,row)
    def neg2pos(self,*cr):
        return tuple( cr_ % cr__ if cr_ < 0 else cr_ for cr_,cr__ in zip(cr,self))
class BlockedMatrixFrame:
    def __init__(self,dim:int):
        self.dim = dim
        self.mat_blocks :List[AllocatableBlock]= []
        self.rhs_blocks :List[AllocatableBlock]= []
        self.crc = ColRowCounter(0,0)
    def add_tricolumn(self,mc:TriRowColumn,):
        a,b,c = mc.sides
        w = mc.width
        prow,pcol = self.crc
        matslctp = tri_column_slices(a,b,c,prow,w,pcol)
        rhsslctp = rhs_column_slices(b,prow)
        ab = AllocatableBlock(mc.matclm,matslctp)
        rhsab = AllocatableBlock(mc.rhsclm,rhsslctp)
        self.mat_blocks.append(ab)
        self.rhs_blocks.append(rhsab)
        self.crc.add(b,w)
        logging.debug(f'Interior block added with rows: {(a,b,c)}, self.crc = {self.crc}')
    def add_block(self,mc:TriRowColumn,upper_left :Tuple[int,int] = (0,0),rhs:bool = False):
        w = mc.width
        h = mc.height
        prow,pcol = self.crc.neg2pos(*upper_left)
        matslctp = (slice(prow,prow +h),slice(pcol,pcol+w))
        # logging.debug(f'upper_left = {upper_left},\t colrow = {pcol,prow}')
        # logging.debug(f'rowslice = {matslctp[0].start,matslctp[0].stop},\t colslice = {matslctp[1].start,matslctp[1].stop}')
        ab = AllocatableBlock(mc.matclm,matslctp)
        self.mat_blocks.append(ab)
        if not rhs:
            return
        rhsslctp = (slice(prow,prow +h),)
        n = len(mc.rhsclm.flatten())
        assert n == h
        rhsab = AllocatableBlock(mc.rhsclm.flatten(),rhsslctp)
        self.rhs_blocks.append(rhsab)
    def extend(self,*stps:int):
        self.crc.move(*stps)
    def move(self,*stps:int):
        for mb in self.mat_blocks:
            mb.move(*stps)
        for mb in self.rhs_blocks:
            mb.move(stps[0])
        self.crc.move(*stps)
        
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
        
        
        

        



