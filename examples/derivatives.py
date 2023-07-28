import numpy as np
from solver.settings import PetrovGalerkinSolverSettings,LinearBoundaryProblem,LinearSolver
import logging
from hybrid.fingerprints import  HybridStateSystem
from hybrid.cramer import CramerRaoBound,BasicLoss
from hybrid.design import ThetaDesign


def evaluate(theta_fun,gradient:bool= False):
    # time = 0.1
    full_edges = True
    hmrf = MatRhsFuns(mode = 'params',theta = theta_fun)
    time = hmrf.total_time
    matfun_, rhsfun_,bndr_cond,edges = hmrf.matfun,hmrf.rhsfun,hmrf.boundary_conditions(),hmrf.edges(full = full_edges)
    pgs = PetrovGalerkinSolverSettings(degree_increments = (4,8,16),\
                    max_rep_err=1e-5,max_lcl_err=1e-4,max_num_interval=2**12,)
    logging.info('\n'+str(pgs))
    

    lbp = LinearBoundaryProblem(funs = (matfun_,rhsfun_),\
                    boundary_condition= bndr_cond,edges = edges)
    ls = LinearSolver(pgs,lbp)
    ls.refine_for_representation()
    ls.refine_for_local_problems()
    
    ls.solve()
    
    
    hss = HybridStateSolution(ls.solution,hmrf.theta_seq,hmrf.trf_seq)
    
    lossfun = BasicLoss(hmrf.fingerprint_edges(),[1,2,3],)
    if not gradient:
        return lossfun(hss),None
    
    hmrf1 = MatRhsFuns(time = time,mode = 'design',design_param='theta1')
    hmrf2 = MatRhsFuns(time = time,mode = 'design',design_param='theta2')

    design1 = ls.solution.matching_gcheb_from_functions(hmrf1.matfun,hmrf1.rhsfun)
    design2 = ls.solution.matching_gcheb_from_functions(hmrf2.matfun,hmrf2.rhsfun)
    
    
    
    thetadeg = ThetaDesign(hss)
    gradfac = ls.get_gradient_factory(lossfun,thetadeg)

    grad1,dldesign = gradfac.get_gradient(hss,design1,ls.global_system_solver)
    grad2,_ = gradfac.get_gradient(hss,design2,ls.global_system_solver)
    
    grad = thetadeg.grad_unite(grad1,grad2,dldesign)
    return lossfun(hss),grad
def main():
    np.random.seed(0)
    theta_fun = np.random.rand(100)*np.pi/2
    ls,gr = evaluate(theta_fun,gradient = True)
    print(ls)
    stepsize = 1e-2
    th1 = theta_fun + stepsize*gr
    ls1,_ = evaluate(th1,gradient = False)
    print(ls,ls1)
if __name__== '__main__':
    main()