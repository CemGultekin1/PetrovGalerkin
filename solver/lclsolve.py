import logging
from typing import Callable
from chebyshev import ChebyshevInterval,FlatListOfFuns,GridwiseChebyshev
from solver.eqgen import LocalEquationFactory
from solver.core import BoundaryCondition
import numpy as np

from solver.glbsys import GlobalSysAllocator
def left_right_decorator(fun:Callable):
    def wrapped_fun(*args,**kwargs):
        left = kwargs.pop('left',False)
        right = kwargs.pop('right',False)
        if left + right != 1:
            logging.error(f'Received left + right = {left} + {right} $\\neq$ 1 ')
            raise Exception
        return fun(*args,left = left,**kwargs)
    return wrapped_fun
class LocalEquationFactory(LocalEquationFactory):
    def __init__(self, leqf:LocalEquationFactory) -> None:
        self.bndr_cond = self.generate_boundary_condition(leqf.dim,)
        self.__dict__.update(leqf.__dict__)
    @classmethod
    def generate_boundary_condition(cls,dim:int):        
        bone = np.eye(dim)
        bzer = np.zeros((dim,dim))
        crhs = np.zeros((dim,))
        return BoundaryCondition(bone,bzer,crhs)

class LocalSysAllocator(GlobalSysAllocator):
    def __init__(self, dim: int, funs: FlatListOfFuns, lcleq: LocalEquationFactory) -> None:
        super().__init__(dim, funs, lcleq)
        self.local_equation =  LocalEquationFactory(lcleq)
    def local_system_blocks(self,chebint:ChebyshevInterval):
        blocks = self.get_single_interval_blocks(chebint)
        return blocks
        
    def get_single_interval_blocks(self,chebint:ChebyshevInterval,):
        gcheb = GridwiseChebyshev.from_single_chebyshev(self.flof,chebint)
        lblocks = self.create_blocks(gcheb)
        return lblocks
        


