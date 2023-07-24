import logging
from typing import Callable
from chebyshev import ChebyshevInterval,FlatListOfFuns,GridwiseChebyshev
from solver.eqgen import EquationFactory
from solver.core import BoundaryCondition
import numpy as np

from solver.glbsys import GlobalSysAllocator
from solver.interior import AdjointInteriorElementFactory
# def left_right_decorator(fun:Callable):
#     def wrapped_fun(*args,**kwargs):
#         left = kwargs.pop('left',False)
#         right = kwargs.pop('right',False)
#         if left + right != 1:
#             logging.error(f'Received left + right = {left} + {right} $\\neq$ 1 ')
#             raise Exception
#         return fun(*args,left = left,**kwargs)
#     return wrapped_fun
class LocalEquationFactory(EquationFactory):
    def __init__(self, leqf:EquationFactory) -> None:
        self.bndr_cond = self.generate_boundary_condition(leqf.dim,)
        self.__dict__.update(leqf.__dict__)
        self.interr = AdjointInteriorElementFactory(leqf.interr)
        
    @classmethod
    def generate_boundary_condition(cls,dim:int):        
        bone = np.eye(dim)
        bzer = np.zeros((dim,dim))
        crhs = np.zeros((dim,))
        return BoundaryCondition(bone,bzer,crhs)

class LocalSysAllocator(GlobalSysAllocator):
    def __init__(self, dim: int,lcleq: EquationFactory) -> None:
        super().__init__(dim, lcleq)
        self.local_equation =  LocalEquationFactory(lcleq)
        
    def local_system_blocks(self,chebint:ChebyshevInterval):
        blocks,rhs = self.get_single_interval_blocks(chebint)
        return blocks,rhs
        
    def get_single_interval_blocks(self,chebint:ChebyshevInterval,):
        gcheb = GridwiseChebyshev.from_single_chebyshev(chebint,chebint)
        lblocks = self.create_blocks(gcheb)
        p = chebint.degree
        x = self.local_equation.interr.dub_quads[:p,:p] * chebint.h/2
        x = np.concatenate([x,x[:2]*0],axis = 0)
        x = x.reshape([p+2,1,p,1])
        rhs = x*np.eye(self.dim).reshape([1,self.dim,1,self.dim])
        rhs = rhs.reshape(((p+2)*self.dim,p*self.dim))
        rhs = rhs
        return lblocks,rhs
        


