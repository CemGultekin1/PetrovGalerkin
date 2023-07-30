from chebyshev import ChebyshevInterval,GridwiseChebyshev
from solver.eqgen import EquationFactory
from solver.bndrcond import BoundaryCondition
import numpy as np
from solver.glbsys import GlobalSysAllocator
from solver.interior import AdjointInteriorElementFactory
class LocalEquationFactory(EquationFactory):
    def __init__(self, leqf:EquationFactory) -> None:        
        self.__dict__.update(leqf.__dict__)
        self.interr = AdjointInteriorElementFactory(leqf.interr)
        leqf.setup_handle.append(self,)
    def setup_for_operations(self, boundary_condition: BoundaryCondition):
        self.bndr_cond = self.bndr.create_boundary_condition_element_factory(boundary_condition)
    @classmethod
    def generate_boundary_condition(cls,dim:int):
        bone = np.eye(dim)
        bzer = np.zeros((dim,dim))
        crhs = np.zeros((dim,))
        return BoundaryCondition(bone,bzer,crhs)

class LocalSysAllocator(GlobalSysAllocator):
    def __init__(self, lcleq: EquationFactory) -> None:
        super().__init__(lcleq)
        self.local_equation =  LocalEquationFactory(lcleq)

        
    def get_single_interval_blocks(self,chebint:ChebyshevInterval,chebint1:ChebyshevInterval):
        gcheb = GridwiseChebyshev.from_single_chebyshev(chebint1,chebint1)
        lblocks = self.create_blocks(gcheb,(chebint.degree,))
        p = chebint.degree
        rhs = np.zeros(((p+2)*self.dim,self.dim))
        rhs[-2*self.dim:-self.dim,:] = np.eye(self.dim)
        rhs[-self.dim:,:] = np.eye(self.dim)
        return lblocks,rhs

