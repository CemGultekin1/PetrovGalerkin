from scipy.sparse.linalg import spsolve
import numpy as np
from .glbsys import SparseGlobalSystem,DenseLocalSystem
from chebyshev import GridwiseChebyshev
class GlobalSystemSolver(SparseGlobalSystem):
    def __init__(self,spglblsys:SparseGlobalSystem) -> None:
        self.__dict__.update(spglblsys.__dict__)
        self.solution = np.empty(0)
    def solve(self,):
        self.solution =  spsolve(self.mat,self.rhs)
        if not isinstance(self.solution,np.ndarray):
            self.solution = self.solution.toarray()
    def get_wrapped_solution(self,gcheb:GridwiseChebyshev)->GridwiseChebyshev:
        return gcheb.create_from_solution(self.solution,self.dim)
        
    
class LocalSystemSolver(DenseLocalSystem):
    def __init__(self,spglblsys:DenseLocalSystem) -> None:
        self.__dict__.update(spglblsys.__dict__)
        self.solution = np.empty(0)
    def solve(self,):
        self.solution =  np.linalg.inv(self.mat)
