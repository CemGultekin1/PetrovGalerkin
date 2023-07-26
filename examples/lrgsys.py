import os
import numpy as np
from fldrsys.folders import OutputsFolders
from solver.settings import PetrovGalerkinSolverSettings,LinearBoundaryProblem,LinearSolver
import logging
from examples.plotter import MultiDimGridwiseChebyshevPlotter
import matplotlib.pyplot as plt



dim = 1
p = np.pi*25
deg = 3
epsl = 1e-7
def fun_g(x):
    return np.cos(p*x**deg)
def fun_dg(x):
    return-np.sin(p*x**deg)*p*x**(deg-1)*deg
def fun_f(x):
    return np.log(x + epsl)* np.sqrt(epsl)
def fun_df(x):
    return 1/(x + epsl )* np.sqrt(epsl)
    
def matfun(x):
    return fun_df(x).reshape([-1,dim,dim])
def rhsfun(x):
    y = np.exp(fun_f(x))*fun_dg(x)
    return y.reshape([-1,dim])#np.stack([x**(i+1) for i in range(dim)],axis = 1).reshape([-1,dim])
def true_solution(x):
    return (np.exp(fun_f(x))*fun_g(x)).reshape([-1,dim])

def main():
    # rlo = RandomLinearODE(dim,)
    matfun_, rhsfun_ = matfun,rhsfun
    # logging.info(f'matfun(0).shape = {matfun(0).shape}')
    # return
    pgs = PetrovGalerkinSolverSettings(degree_increments = (4,8,16,),\
                    max_rep_err=1e-12,max_lcl_err=1e-12,max_num_interval=2**12,)
    logging.info('\n'+str(pgs))
    
    eyemat = np.eye(dim)
    b0,b1 = eyemat,eyemat
    c = b1*true_solution(1) + b0*true_solution(0)
    
    lbp = LinearBoundaryProblem(funs = (matfun_,rhsfun_),boundary_condition= (b0,b1,c))
    ls = LinearSolver(pgs,lbp)
    ls.refine_for_representation()
    foldername = OutputsFolders().from_file_name(__file__).create().to_str()    
    gcp = MultiDimGridwiseChebyshevPlotter()
    for i,(fig,axs) in enumerate(gcp.draw(ls.mergedfuns)):  
        axs.set_title(i)      
        filename = f'repref_{i}.png'
        plt.savefig(os.path.join(foldername,filename))
        plt.close()
    ls.refine_for_local_problems()
    for i,(fig,axs) in enumerate(gcp.draw(ls.solution)):  
        axs.set_title(i)      
        filename = f'slvref_{i}.png'
        plt.savefig(os.path.join(foldername,filename))
        plt.close()
        
    ls.solve()
    
    for i,(fig,axs) in enumerate(gcp.draw(ls.solution,true_solution)):  
        axs.set_title(i)      
        filename = f'sltn_{i}.png'
        plt.savefig(os.path.join(foldername,filename))
        plt.close()
if __name__== '__main__':
    main()