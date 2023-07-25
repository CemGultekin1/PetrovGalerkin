import os
import numpy as np
from fldrsys.folders import OutputsFolders
from solver.settings import PetrovGalerkinSolverSettings,LinearBoundaryProblem,LinearSolver
import logging
from examples.plotter import MultiDimGridwiseChebyshevPlotter
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)


dim = 1
def matfun(x):
    p = 4
    return np.stack([np.cos(np.exp(x*p)/np.exp(p/2)*np.pi)*(1+p*x)**2 for i in range(dim**2)],axis = 1).reshape([-1,dim,dim])
def rhsfun(x):
    return np.stack([x**(i+1) for i in range(dim)],axis = 1).reshape([-1,dim])
def main():
    pgs = PetrovGalerkinSolverSettings(degree_increments = (1,2,4,6,8),max_rep_err=1e-3,max_lcl_err=1e-7)
    logging.info('\n'+str(pgs))
    b0,b1,c = np.eye(dim),np.zeros((dim,dim)),np.zeros((dim,1))
    lbp = LinearBoundaryProblem(funs = (matfun,rhsfun),boundary_condition= (b0,b1,c))
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
if __name__== '__main__':
    main()