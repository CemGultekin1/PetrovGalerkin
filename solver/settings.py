from dataclasses import dataclass
import logging
from typing import Tuple
import numpy as np
from chebyshev import NumericFunType,ListOfFuns,GridwiseChebyshev,RepresentationRefiner
from solver.bndrcond import BoundaryCondition
from solver.eqgen import EquationFactory
from solver.errcontrol import SolutionRefiner
from .lclerr import LocalErrorEstimate

@dataclass
class PetrovGalerkinSolverSettings:
    max_num_interval :float = 2**8
    max_rep_err : float = 1e-3
    max_lcl_err : float = 1e-3
    max_grid_cond :float = 64.
    degree_increments :Tuple[int] = (2,)
    def __str__(self,):
        def strfun(val):
            return "{:.1e}".format(val) if isinstance(val,float) else str(val)
        keylist =self.__annotations__.keys()
        maxlen = max([len(x) for x in keylist])
        titles = []
        for key in keylist:
            clen = len(key)
            pad =maxlen - clen
            titles.append(key + pad*' ' + ':\t')
        for i,(key,ttl) in enumerate(zip(keylist,titles)):
            val = self.__getattribute__(key)
            if np.isscalar(val):
                valstr = strfun(val)
            else:
                valstr ='('+ ','.join([strfun(val_) for val_ in val]) + ')'
            titles[i] = ttl + valstr
        return "\n".join(titles)
    @property
    def max_degree(self,):
        return self.degree_increments[-1]
    @property
    def min_degree(self,):
        return self.degree_increments[0]
        
            
class LinearBoundaryProblem:
    def __init__(self,\
            funs:Tuple[NumericFunType,NumericFunType] = (),\
            boundary_condition: Tuple[np.ndarray,np.ndarray,np.ndarray] = (np.empty(0,),np.empty(0,),np.empty(0)),\
            boundaries :Tuple[float,float] = (0.,1.),) -> None:
        self.matfun, self.rhsfun = funs
        self.boundary_condition = BoundaryCondition(*boundary_condition)
        self.boundaries = boundaries
        self.dim = self.boundary_condition.dim
    
class LinearSolver(LinearBoundaryProblem,PetrovGalerkinSolverSettings):
    def __init__(self,pgs:PetrovGalerkinSolverSettings, lbp:LinearBoundaryProblem)-> None:
        self.__dict__.update(lbp.__dict__)  
        self.__dict__.update(pgs.__dict__)        
        self.listfuns = ListOfFuns(self.matfun,self.rhsfun)
        self.flatlistfuns = self.listfuns.flatten()        
        self.mergedfuns = GridwiseChebyshev.from_function(self.flatlistfuns,self.min_degree,*lbp.boundaries)
        self.equfactory  = EquationFactory(self.dim,pgs.max_degree,self.boundary_condition)
        self.lclerr = LocalErrorEstimate(self.dim,self.equfactory)
        self.repref = RepresentationRefiner(self.degree_increments,self.max_rep_err,self.max_num_interval,self.max_grid_cond)
        self.solref = SolutionRefiner(self.lclerr,self.degree_increments,\
            self.max_lcl_err,self.max_num_interval,self.max_grid_cond)
        self.solution = None
    def refine_for_local_problems(self,):
        self.solution = self.mergedfuns.new_grided_chebyshev(self.dim,degree = self.degree_increments[0])
        max_iter_num = 128
        for i in range(max_iter_num):
            flag = self.solref.run_controls(self.solution,self.mergedfuns)
            if flag:
                break
            self.solref.run_refinements(self.solution,self.mergedfuns)
        self.mergedfuns.update_edge_values()
        self.solution.update_edge_values()
    def refine_for_representation(self,):
        max_iter_num = 128
        for i in range(max_iter_num):            
            flag = self.repref.run_controls(self.mergedfuns)
            if flag:
                break
            self.repref.run_refinements(self.mergedfuns)
        self.mergedfuns.update_edge_values()