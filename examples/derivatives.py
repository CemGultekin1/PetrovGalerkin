import numpy as np
from solver.settings import PetrovGalerkinSolverSettings,LinearBoundaryProblem,LinearSolver
import logging
from hybrid.fingerprints import  Fingerprints, HybridStateSystem,HybridStateSolution
from hybrid.cramer import CramerRaoBound
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
logging.basicConfig(level=logging.INFO)

        
def evaluate(theta_fun:np.ndarray,grad_flag:bool =False):    
    hmrf = HybridStateSystem(mode = 'params',theta = theta_fun)
    matfun_, rhsfun_,bndr_cond,edges = hmrf.matfun,hmrf.rhsfun,hmrf.boundary_conditions(),hmrf.starting_edges
    pgs = PetrovGalerkinSolverSettings(degree_increments = (1,),\
                    max_rep_err=1e-5,max_lcl_err=1e-4,max_num_interval=2**12,)
    
    lbp = LinearBoundaryProblem(funs = (matfun_,rhsfun_),\
                    boundary_condition= bndr_cond,edges = edges)
    ls = LinearSolver(pgs,lbp) 
    ls.solve()
    
    solution = ls.solution
    hss = HybridStateSolution(solution,theta_fun,hmrf.trf_seq)
    fng = Fingerprints(hss)
    crb = CramerRaoBound(hss.fingerprint_edges(),[1,2,3])
    
    
    adjoint_method = ls.adjoint_method()
    design_product = ls.design_product()
    
    hmrf1 = HybridStateSystem(mode = 'design',design_param='theta1',theta = theta_fun)
    design1 = ls.solution.matching_gcheb_from_functions(hmrf1.matfun,hmrf1.rhsfun)
    
    hmrf2 = HybridStateSystem(mode = 'design',design_param='theta2',theta = theta_fun)
    design2 = ls.solution.matching_gcheb_from_functions(hmrf2.matfun,hmrf2.rhsfun)
    
    l0 = crb(fng)
    

    if not grad_flag:
        return l0,np.empty((0,))
    
    
    
    dldf = crb.gradient(fng)
    dldedge = fng.state_edges_derivative_inner_product(dldf)    
    dldf_dfdtheta = fng.design_derivative_inner_product(dldf)
    adjoint = adjoint_method.get_adjoint(solution,ls.global_system_solver,dldedge)
    
    dldx1 = design_product.dot(solution,adjoint,design1)
    dldx2 = design_product.dot(solution,adjoint,design2)
    dldx = hmrf.design_gradient_collection(dldx1,dldx2,dldf_dfdtheta,solution)
    return l0,dldx

    

def main():
    np.random.seed(0)
    theta_fun = np.random.rand(100)*np.pi/2
    l0,grad = evaluate(theta_fun,grad_flag = True)
    pert = np.random.randn(*theta_fun.shape)*0
    pert[0] = 1
    for h in [10**(-i) for i in range(1,15)]:        
        theta_fun += h*pert
        l1,_ = evaluate(theta_fun,grad_flag = False)
        theta_fun -= h*pert
        
        g_dl = pert.flatten()@grad.flatten()
        t_dl = (l1 - l0)/h
        
        relerr = np.abs(
            g_dl - t_dl
        )/(np.abs(g_dl) + np.abs(t_dl))*2
        logging.info(f'relerr,h = {relerr,h}')
    
    
    


if __name__== '__main__':
    main()