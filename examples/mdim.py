import os
import numpy as np
from fldrsys.folders import OutputsFolders
from solver.settings import PetrovGalerkinSolverSettings,LinearBoundaryProblem,LinearSolver
import logging
from examples.plotter import MultiDimGridwiseChebyshevPlotter
import matplotlib.pyplot as plt
from scipy.linalg import expm


dim = 10
np.random.seed(0)
mat = np.random.randn(dim,dim)
# mat = (mat.T + mat)/2
rhs = np.random.randn(dim,)
y0 = np.random.randn(dim,)
# mat = np.eye(dim) * np.random.randn(1,)
# rhs = np.random.randn(dim,)
 
def matfun(x):
    return np.stack([mat]*len(x),axis = 0).reshape(len(x),dim,dim)
def rhsfun(x):
    y = np.stack([rhs]*len(x),axis = 0)
    return y.reshape([-1,dim])#np.stack([x**(i+1) for i in range(dim)],axis = 1).reshape([-1,dim])
def true_solution(x,):
    if np.isscalar(x):
        return true_solution(np.array([x]))
    def singular_matrix(x_):
        return np.linalg.inv(mat)@(expm(mat*x_) - np.eye(dim) )@rhs  + expm(mat*x_)@y0
    return np.stack(list(map(singular_matrix,x)),axis = 0)

def main():
    matfun_, rhsfun_ = matfun,rhsfun
    pgs = PetrovGalerkinSolverSettings(degree_increments = (2,4,6,8),\
                    max_rep_err=1e-2,max_lcl_err=1e-8,max_num_interval=2**12,)
    logging.info('\n'+str(pgs))
    
    eyemat = np.eye(dim)
    b0,b1 = eyemat,eyemat
    c = b0@true_solution(0).reshape([-1]) + b1@true_solution(1).reshape([-1])
    
    lbp = LinearBoundaryProblem(funs = (matfun_,rhsfun_),boundary_condition= (b0,b1,c))
    ls = LinearSolver(pgs,lbp)
    # ls.refine_for_representation()
    foldername = OutputsFolders().from_file_name(__file__).create().to_str()    
    gcp = MultiDimGridwiseChebyshevPlotter()
    
    ls.refine_for_local_problems()
    
    ls.solve()
    
    for i,(fig,axs) in enumerate(gcp.draw(ls.solution,true_solution)):  
        axs.set_title(i)      
        filename = f'sltn_{i}.png'
        plt.savefig(os.path.join(foldername,filename))
        plt.close()
if __name__== '__main__':
    main()