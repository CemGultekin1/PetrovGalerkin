from dataclasses import dataclass
import logging
from typing import  List, Tuple
from .interval import GridwiseChebyshev,ChebyshevInterval
from .funs import NumericFunType,ListOfFuns,FlatListOfFuns
from .interpolate import ErrorEstimator
import numpy as np
def is_square(x:np.ndarray):
    if x.ndim != 2:
        return False
    if x.shape[0] != x.shape[1]:
        return False
    return True
def same_shape(x1:np.ndarray,x2:np.ndarray):
    return x1.shape == x2.shape
class BoundaryCondition:
    B0:np.ndarray
    B1:np.ndarray
    c:np.ndarray
    def __init__(self,b0:np.ndarray,b1:np.ndarray,c:np.ndarray) -> None:
        self.B0 = b0
        self.B1 = b1
        self.c = c
        try:
            if (not is_square(b0) ) or (not is_square(b1)) or (not same_shape(b0,b1)):
                raise Exception
            if b0.shape[0] != c.shape[0]:
                raise Exception
        except:
            logging.error(f'Boundary condition dimensions are not consistent: {b0.shape},{b1.shape},{c.shape}' )
            raise Exception

    def adjoint(self,)->'BoundaryCondition':...
    @property
    def dim(self,):
        return self.B0.shape[0]

class LinearBoundaryProblem:
    def __init__(self,\
            funs:Tuple[NumericFunType,NumericFunType] = (),\
            boundary_condition: Tuple[np.ndarray,np.ndarray] = (np.empty(0,),np.empty(0,)),\
            boundaries :Tuple[float,float] = (0.,1.),) -> None:
        self.matfun, self.rhsfun = funs
        self.boundary_condition = BoundaryCondition(*boundary_condition)
        self.boundaries = boundaries
        self.dim = self.boundary_condition.dim
    
@dataclass
class PetrovGalerkinSolverSettings:
    max_element_num :int = int(2**8)
    grid_regularity_bound :float = 64.
    min_degree:int = 3
    max_degree:int = 8


class LinearSolver(LinearBoundaryProblem):
    def __init__(self,pgs:PetrovGalerkinSolverSettings, lbp:LinearBoundaryProblem)-> None:
        self.dim = lbp.dim
        self.solver_setttings = pgs
        self.__dict__.update(lbp.__dict__)        
        self.listfuns = ListOfFuns(self.matfun,self.rhsfun)
        self.flatlistfuns = self.listfuns.flatten()        
        self.mergedfuns = GridwiseChebyshev(self.flatlistfuns,*lbp.boundaries)
        
        