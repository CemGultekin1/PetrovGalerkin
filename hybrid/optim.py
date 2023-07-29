from copy import deepcopy
import logging
from typing import Dict, List
from hybrid.cramer import CramerRaoBound
from hybrid.handles import DesignParameteric, Parameteric
from solver.settings import LinearSolver,LinearBoundaryProblem,PetrovGalerkinSolverSettings
from hybrid.fingerprints import HybridStateSystem,HybridStateSolution,Fingerprints
from hybrid.eqtns import DesignSequences
import numpy as np


class HybridStateSystemsForOptimization(Parameteric,DesignParameteric):
    param_hss:HybridStateSystem 
    design_theta1_hss:HybridStateSystem 
    design_theta2_hss:HybridStateSystem 
    def __init__(self,**kwargs) -> None:
        self.org_hss = HybridStateSystem(mode = 'org', **kwargs)
        self.param_hss = HybridStateSystem(mode = 'params', **kwargs)
        self.design_theta1_hss = HybridStateSystem(mode = 'design', design_param='theta1',**kwargs)
        self.design_theta2_hss = HybridStateSystem(mode = 'design', design_param='theta2',**kwargs)
        self.org_linear_boundary_problem,self.params_linear_boundary_problem = self.create_linear_boundary_problem()
    def create_linear_boundary_problem(self,):
        matfun_, rhsfun_,bndr_cond,edges = self.param_hss.matfun,\
                        self.param_hss.rhsfun,\
                        self.param_hss.boundary_conditions(),\
                        self.param_hss.starting_edges      
            
        params_bp  = LinearBoundaryProblem(funs = (matfun_,rhsfun_),\
                        boundary_condition= bndr_cond,edges = edges)
        
        matfun_, rhsfun_,bndr_cond,edges = self.org_hss.matfun,\
                        self.org_hss.rhsfun,\
                        self.org_hss.boundary_conditions(),\
                        self.org_hss.starting_edges    
        org_bp  = LinearBoundaryProblem(funs = (matfun_,rhsfun_),\
                        boundary_condition= bndr_cond,edges = edges)
        return org_bp,params_bp
    def change_of_parameters(self,**kwargs):
        super().change_of_parameters(**kwargs)
        self.org_linear_boundary_problem,self.params_linear_boundary_problem = self.create_linear_boundary_problem()
    def change_of_design_parameters(self,dsecs:DesignSequences):
        return super().change_of_design_parameters(theta_seq = dsecs.theta_seq,trf_seq = dsecs.trf_seq)
    @property
    def theta_seq(self,):
        return self.param_hss.theta_seq
    @property
    def trf_seq(self,):
        return self.param_hss.trf_seq



class HybridStateJacobians(HybridStateSystemsForOptimization):
    def __init__(self,linear_solver:LinearSolver,**kwargs) -> None:
        self.linear_solver = linear_solver
        super().__init__(**kwargs)
        self.reset()
    def reset(self,):
        self._org_solution = None
        self._params_solution = None
    def change_of_design_parameters(self, *args):
        self.reset()
        return super().change_of_design_parameters(*args)
    def change_of_parameters(self, **kwargs):
        self.reset()
        return super().change_of_parameters(**kwargs)
    @property
    def org_solution(self,):
        if self._org_solution is None:
            solution,gss = self.linear_solver.solve(self.org_linear_boundary_problem)
            hssol = HybridStateSolution(solution,self.theta_seq,self.trf_seq,gss, self.org_hss)
            self._org_solution = hssol
        return self._org_solution
    @property
    def params_solution(self,):
        if self._params_solution is None:
            solution,gss = self.linear_solver.solve(self.params_linear_boundary_problem)
            hssol = HybridStateSolution(solution,self.theta_seq,self.trf_seq,gss, self.param_hss)
            self._params_solution = hssol
        return self._params_solution
    
    @property
    def org_fingerprints(self,):
        sol = self.org_solution
        return Fingerprints(sol)
        
    @property
    def params_fingerprints(self,):
        sol = self.params_solution
        return Fingerprints(sol)
    
    @property
    def optimal_design_loss(self,):
        fng = self.params_fingerprints
        crb = CramerRaoBound(fng.fingerprint_edges(),[0,1,2,])
        return crb(fng)
    @property
    def optimal_design_jacobian(self,):
        sol = self.params_solution
        fng = self.params_fingerprints
        
        adjoint_method = self.linear_solver.adjoint_method()
        design_product = self.linear_solver.design_product()
        
        design1 = sol.matching_gcheb_from_functions(self.design_theta1_hss.matfun,self.design_theta1_hss.rhsfun)
        design2 = sol.matching_gcheb_from_functions(self.design_theta2_hss.matfun,self.design_theta2_hss.rhsfun)        
       
        crb = CramerRaoBound(fng.fingerprint_edges(),[0,1,2,])
        dldf = crb.gradient(fng)
        dldedge = fng.state_edges_derivative_inner_product(dldf)    
        dldf_dfdtheta = fng.design_derivative_inner_product(dldf)
        adjoint = adjoint_method.get_adjoint(sol,sol.global_sys_sol,dldedge)
        
        dldx1 = design_product.dot(sol,adjoint,design1)
        dldx2 = design_product.dot(sol,adjoint,design2)
        dldx = self.param_hss.design_gradient_collection(dldx1,dldx2,dldf_dfdtheta,sol)
        return dldx
        
    
    

class OptimalDesign(HybridStateJacobians):
    def __init__(self, linear_solver: LinearSolver = LinearSolver(PetrovGalerkinSolverSettings()), **kwargs) -> None:
        super().__init__(linear_solver, **kwargs)
    def update(self,*params):
        ds = DesignSequences(theta_seq=params[0],trf_seq=self.trf_seq)
        self.change_of_design_parameters(ds)
    def jac(self,*params):        
        self.update(*params)
        return self.optimal_design_jacobian
    def eval(self,*params):
        self.update(*params)
        return self.optimal_design_loss
    



class NLLS(HybridStateJacobians):
    true_solution : HybridStateSolution
    corrupted_fingerprints : Fingerprints
    def __init__(self,  linear_solver: LinearSolver = LinearSolver(PetrovGalerkinSolverSettings()), **kwargs) -> None:        
        super().__init__(linear_solver, **kwargs)
        self.true_solution = None
        self.corrupted_fingerprints = None
    def update(self,*params):
        curparams = dict(
            zip(
                self.param_hss.solved_param_names,params
            )
        )
        self.change_of_parameters(**curparams)

    def set_true_solution(self,*params,**kwargs):
        if not bool(kwargs):
            self.update(*params)
        else:
            pd = deepcopy(self.param_hss.params_dict)
            pd.update(kwargs)
            self.update(*list(pd.values()))
        self.true_solution = self.org_solution
    def corrupt_with_gaussian_noise(self,sigma:float):
        vals = self.true_fingerprints.values
        noise = np.random.randn(*vals.shape)*sigma
        self.corrupted_fingerprints = vals  + noise 
        
    @property
    def true_fingerprints(self,):
        if self.true_solution is None:
            raise Exception
        return Fingerprints(self.true_solution)
    
    
    @property
    def true_params(self,):
        if self.true_solution is None:
            raise Exception
        return np.array(list(self.true_solution.params_dict.values()))
    def jac(self,*params,**kwargs)->np.ndarray:
        if not bool(kwargs):
            self.update(*params)
        else:
            pd = deepcopy(self.param_hss.params_dict)
            pd.update(kwargs)
            self.update(*list(pd.values()))
        fng =  self.org_fingerprints.values
        tfng = self.corrupted_fingerprints.values
        pfng  = self.params_fingerprints.values
        err = fng - tfng
        grad :np.ndarray = err @ pfng
        assert grad.size == len(params)
        return grad
    
    def eval(self,*params,**kwargs)->float:
        if not bool(kwargs):
            self.update(*params)
        else:
            pd = deepcopy(self.param_hss.params_dict)
            pd.update(kwargs)
            self.update(*list(pd.values()))
        fng =  self.org_fingerprints.values
        tfng = self.corrupted_fingerprints
        return np.sum((fng - tfng)**2)
    
        
        