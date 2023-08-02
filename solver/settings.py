from dataclasses import dataclass
import logging
from typing import Tuple
import numpy as np
from chebyshev import NumericFunType,ListOfFuns,GridwiseChebyshev,RepresentationRefiner
from solver.bndrcond import BoundaryCondition
from solver.dimensional import Dimensional
from solver.eqgen import EquationFactory
from solver.errcontrol import SolutionRefiner
from solver.glbsys import GlobalSysAllocator, SparseGlobalSystem
from .lclerr import LocalErrorEstimate
from .linsolve import GlobalSystemSolver
from solver.design import AdjointMethod, DesignProduct, TimeInstanceDesignProduct
    
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
        
BoundaryConditionType = Tuple[np.ndarray,np.ndarray,np.ndarray]
class LinearBoundaryProblem:
    def __init__(self,\
            funs:Tuple[NumericFunType,NumericFunType] = (),\
            boundary_condition: BoundaryConditionType = (np.empty((0,0)),np.empty((0,0)),np.empty(0)),\
            edges :Tuple[float,...] = (0.,1.),) -> None:
        self.matfun, self.rhsfun = funs
        self.boundary_condition = BoundaryCondition(*boundary_condition)
        self.edges = edges
        self.dim = self.boundary_condition.dim
    def setup_boundary_condition(self,boundary_condition:BoundaryConditionType):
        self.boundary_condition = BoundaryCondition(*boundary_condition)
        
class LinearSolver(PetrovGalerkinSolverSettings,Dimensional):
    def __init__(self,pgs:PetrovGalerkinSolverSettings, )-> None:
        Dimensional.__init__(self,)
        self.__dict__.update(pgs.__dict__)                       
        self.equfactory  = EquationFactory(pgs.max_degree,)
        self.lclerr = LocalErrorEstimate(self.equfactory)
        self.repref = RepresentationRefiner(self.degree_increments,self.max_rep_err,self.max_num_interval,self.max_grid_cond)
        self.solref = SolutionRefiner(self.lclerr,self.degree_increments,\
            self.max_lcl_err,self.max_num_interval,self.max_grid_cond)
        self.global_sys_allocator = GlobalSysAllocator(self.equfactory)
    def solve(self,lbp:LinearBoundaryProblem):
        dim = lbp.dim
        self.set_dim(dim)
        self.equfactory.setup_for_operations(lbp.boundary_condition)
        
        listfuns = ListOfFuns(lbp.matfun,lbp.rhsfun)
        flatlistfuns = listfuns.flatten()        
        if len(lbp.edges)>2:
            mergedfuns = GridwiseChebyshev.from_function_and_edges(flatlistfuns,self.min_degree,lbp.edges)            
        elif len(lbp.edges) == 2:
            mergedfuns = GridwiseChebyshev.from_function(flatlistfuns,self.min_degree,*lbp.edges)
        mergedfuns = self.refine_for_representation(mergedfuns)
        solution = self.refine_for_local_problems(mergedfuns)

        blocks = self.global_sys_allocator.create_blocks(mergedfuns,tuple(solution.ps))
        gss = GlobalSystemSolver(blocks)
        gss.solve()
        solution = gss.get_wrapped_solution(solution,inplace = True)
        return solution,gss
        

    def refine_for_local_problems(self,mergedfuns:GridwiseChebyshev):
        solution = mergedfuns.new_grided_chebyshev(self.dim,degree = self.degree_increments[0])
        max_iter_num = 128
        refinement_happened_flag = False
        for i in range(max_iter_num):
            flag = self.solref.run_controls(solution,mergedfuns)
            if flag:
                break
            self.solref.run_refinements(solution,mergedfuns)
            refinement_happened_flag = True
        if refinement_happened_flag:
            mergedfuns.update_edge_values()
        dims = [solution.cheblist[i].dim for i in range(solution.num_interval)]
        dims = np.array(dims)
        if not np.all(dims==dims[0]):
            logging.error(f'{dims.tolist()}')
            dims = [mergedfuns.cheblist[i].dim for i in range(mergedfuns.num_interval)]
            logging.error(f'{dims}')
            raise Exception
        solution.update_edge_values()
        return solution
    def refine_for_representation(self,mergedfuns:GridwiseChebyshev):
        max_iter_num = 256
        for _ in range(max_iter_num):            
            flag = self.repref.run_controls(mergedfuns)
            if flag:
                break
            self.repref.run_refinements(mergedfuns)
        mergedfuns.update_edge_values()
        return mergedfuns
        
    def adjoint_method(self,):
        return AdjointMethod(self.equfactory,self.dim)
    def design_product(self,):
        return DesignProduct(self.lclerr.lcl_sys_alloc)
    def time_instance_design_product(self,):
        return TimeInstanceDesignProduct(self.lclerr.lcl_sys_alloc)