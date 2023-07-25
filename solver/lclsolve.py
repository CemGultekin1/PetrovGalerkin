from chebyshev import ChebyshevInterval,GridwiseChebyshev
from solver.eqgen import EquationFactory
from solver.bndrcond import BoundaryCondition
import numpy as np
from solver.glbsys import GlobalSysAllocator
from solver.interior import AdjointInteriorElementFactory
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
        
    def get_single_interval_blocks(self,chebint:ChebyshevInterval,problem_components:ChebyshevInterval):
        gcheb = GridwiseChebyshev.from_single_chebyshev(problem_components,problem_components)
        lblocks = self.create_blocks(gcheb,(chebint.degree,))
        p = chebint.degree
        rhs = np.zeros(((p+2)*self.dim,self.dim))
        rhs[-2*self.dim:-self.dim,:] = np.eye(self.dim)
        rhs[-self.dim:,:] = np.eye(self.dim)
        return lblocks,rhs
        


