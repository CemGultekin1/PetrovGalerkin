import logging
from typing import Tuple
from scipy.sparse.linalg import spsolve
import numpy as np

from solver.matalloc import BlockedMatrixFrame
from .glbsys import SparseGlobalSystem,DenseLocalSystem
from chebyshev import GridwiseChebyshev
class GlobalSystemSolver(SparseGlobalSystem):
    def __init__(self,blocks:BlockedMatrixFrame) -> None:
        super().__init__(blocks)
        self.solution = np.empty(0)
    def solve(self,):
        if self.solution.size == 0:
            self.solution =  spsolve(self.mat,self.rhs)
        if not isinstance(self.solution,np.ndarray):
            self.solution = self.solution.toarray()
    def get_wrapped_solution(self,gcheb:GridwiseChebyshev,inplace:bool = False)->GridwiseChebyshev:
        if inplace:
            gcheb.adopt_solution(self.solution,self.dim)
            return gcheb
        return gcheb.create_from_solution(self.solution,self.dim)
    def adjoint_system(self,):
        gss = GlobalSystemSolver.__new__(GlobalSystemSolver,)
        gss.__dict__.update(self.__dict__)
        gss.mat = -gss.mat.transpose()
        gss.solution = np.empty(0)
        return gss
        
        
    
class LocalSystemSolver(DenseLocalSystem):
    def __init__(self,spglblsys:DenseLocalSystem) -> None:
        self.__dict__.update(spglblsys.__dict__)
        self.solution = np.empty(0)
    def solve(self,):
        if self.solution.size == 0:
            self.solution =  -np.linalg.solve(self.mat,self.rhs)
    @property
    def interior_solution(self,):
        if self.solution.size == 0:
            logging.error('System is not solved yet')
            raise Exception
        return self.solution[self.dim:-self.dim,:]
    @property
    def edge_solution(self,)->Tuple[np.ndarray,np.ndarray]:
        return self.solution[:self.dim,:],self.solution[-self.dim:,:]
    
        