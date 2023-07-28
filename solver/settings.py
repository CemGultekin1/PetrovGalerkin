from dataclasses import dataclass
import logging
from typing import Tuple
import numpy as np
from chebyshev import NumericFunType,ListOfFuns,GridwiseChebyshev,RepresentationRefiner
from solver.bndrcond import BoundaryCondition
from solver.eqgen import EquationFactory
from solver.errcontrol import SolutionRefiner
from solver.glbsys import GlobalSysAllocator, SparseGlobalSystem
from .lclerr import LocalErrorEstimate
from .linsolve import GlobalSystemSolver
from solver.design import AdjointMethod, DesignProduct
    
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
            edges :Tuple[float,...] = (0.,1.),) -> None:
        self.matfun, self.rhsfun = funs
        self.boundary_condition = BoundaryCondition(*boundary_condition)
        self.edges = edges
        self.dim = self.boundary_condition.dim
    
class LinearSolver(LinearBoundaryProblem,PetrovGalerkinSolverSettings):
    def __init__(self,pgs:PetrovGalerkinSolverSettings, lbp:LinearBoundaryProblem)-> None:
        self.__dict__.update(lbp.__dict__)  
        self.__dict__.update(pgs.__dict__)        
        self.listfuns = ListOfFuns(self.matfun,self.rhsfun)
        self.flatlistfuns = self.listfuns.flatten()        
        if len(lbp.edges)>2:
            self.mergedfuns = GridwiseChebyshev.from_function_and_edges(self.flatlistfuns,self.min_degree,lbp.edges)            
        elif len(lbp.edges) == 2:
            self.mergedfuns = GridwiseChebyshev.from_function(self.flatlistfuns,self.min_degree,*lbp.edges)
        self.equfactory  = EquationFactory(self.dim,pgs.max_degree,self.boundary_condition)
        self.lclerr = LocalErrorEstimate(self.dim,self.equfactory)
        self.repref = RepresentationRefiner(self.degree_increments,self.max_rep_err,self.max_num_interval,self.max_grid_cond)
        self.solref = SolutionRefiner(self.lclerr,self.degree_increments,\
            self.max_lcl_err,self.max_num_interval,self.max_grid_cond)
        self.solution = self.mergedfuns
        self.global_system_solver = None
    def generate_empty_solution(self,):
        return self.mergedfuns.new_grided_chebyshev(self.dim,degree = self.degree_increments[0])
    def refine_for_local_problems(self,):
        self.solution = self.generate_empty_solution()
        max_iter_num = 128
        for i in range(max_iter_num):
            flag = self.solref.run_controls(self.solution,self.mergedfuns)
            if flag:
                break
            self.solref.run_refinements(self.solution,self.mergedfuns)
        self.mergedfuns.update_edge_values()
        dims = [self.solution.cheblist[i].dim for i in range(self.solution.num_interval)]
        dims = np.array(dims)
        if not np.all(dims==dims[0]):
            logging.error(f'{dims.tolist()}')
            dims = [self.mergedfuns.cheblist[i].dim for i in range(self.mergedfuns.num_interval)]
            logging.error(f'{dims}')
            raise Exception
        self.solution.update_edge_values()
    def refine_for_representation(self,):
        max_iter_num = 256
        for _ in range(max_iter_num):            
            flag = self.repref.run_controls(self.mergedfuns)
            if flag:
                break
            self.repref.run_refinements(self.mergedfuns)
        self.mergedfuns.update_edge_values()
    def solve(self,):
        nags = GlobalSysAllocator(self.dim,self.equfactory)

        blocks = nags.create_blocks(self.mergedfuns,tuple(self.solution.ps))
        sgs = SparseGlobalSystem(blocks)
    
        gss = GlobalSystemSolver(sgs)
        gss.solve()
        self.global_system_solver = gss
        self.solution = gss.get_wrapped_solution(self.solution,inplace = True)
    def adjoint_method(self,):
        return AdjointMethod(self.equfactory,self.dim)
    def design_product(self,):
        return DesignProduct(self.lclerr.lcl_sys_alloc)