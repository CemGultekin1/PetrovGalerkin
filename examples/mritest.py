import os
import numpy as np
from fldrsys.folders import OutputsFolders
from solver.settings import PetrovGalerkinSolverSettings,LinearBoundaryProblem,LinearSolver
import logging
from examples.plotter import MultiDimGridwiseChebyshevPlotter
import matplotlib.pyplot as plt
from hbridstt.eqtns import HybridStateSystem





class MatRhsFuns(HybridStateSystem):
    def __init__(self,num:int = -1,time:float = -1.,mode:str = 'org',design_param :str = 'theta') -> None:
        if time > 0:
            n = np.floor(time/self.tr).astype(int)
        elif num > 0:
            n = num
        else:
            raise Exception
        self.design_param = design_param
        self.mode = mode
        np.random.seed(1)
        alphas = np.random.randn(n)
        alphas = np.cumsum(alphas)/ np.cumsum(alphas*0 + 1)
        alphas = alphas*np.pi/2 + np.pi/6
        trfs = np.random.rand(n-1)*1e-3
        super().__init__(alphas,trfs)
        self.dim = 2
    @property
    def signal_names(self,):
        snms = super().signal_names
        if self.mode == 'org':
            return snms
        prms = super().param_names
        return [f'd{snm}/d{prm}' for snm in snms for prm in prms]
    def edges(self,):
        return tuple((self.tr*np.arange(self.num_int)).tolist())
    def fingerprint_edges(self,):
        return self.edges()[1:-1]
        
    def matfun(self,x,):
        if self.mode == 'org':
            return self.org_sys_mat(x)
        elif self.mode == 'params':
            return self.params_sys_mat(x)
        elif self.mode == 'design':
            return self.design_sys_mat(x,name = self.design_param)
    def rhsfun(self,x,):
        if self.mode == 'org':
            return self.org_sys_rhs(x)
        elif self.mode == 'params':
            return self.params_sys_rhs(x)
        elif self.mode == 'design':
            return self.design_sys_rhs(x,name = self.design_param)
    def boundary_conditions(self,):
        if self.mode == 'org':
            return self.org_sys_bndr()
        elif self.mode == 'params':
            return self.params_sys_bndr()
        elif self.mode == 'design':
            return  self.design_sys_bndr(name = self.design_param)


def main():
    hmrf = MatRhsFuns(time = 0.5,mode = 'params')
    matfun_, rhsfun_,bndr_cond,edges = hmrf.matfun,hmrf.rhsfun,hmrf.boundary_conditions(),hmrf.edges()
    pgs = PetrovGalerkinSolverSettings(degree_increments = (8,12,16,),\
                    max_rep_err=1e-2,max_lcl_err=1e-3,max_num_interval=2**12,)
    logging.info('\n'+str(pgs))
    

    lbp = LinearBoundaryProblem(funs = (matfun_,rhsfun_),\
                    boundary_condition= bndr_cond,edges = edges)
    ls = LinearSolver(pgs,lbp)
    ls.refine_for_representation()
    ls.refine_for_local_problems()
    
    ls.solve()
    
    
    hmrf = MatRhsFuns(time = 3,mode = 'design')
    dtheta = ls.solution.matching_gcheb_from_functions(hmrf.matfun,hmrf.rhsfun,)
    
    
    return
    x = np.array(hmrf.fingerprint_edges())
    y = ls.solution(x)
    
    
    
    fig,axs = plt.subplots(1,1,figsize = (7,5))
    axs.plot(x,y,linewidth = 1,label = hmrf.signal_names)
    axs.legend()
    plt.savefig('dummy.png')
if __name__== '__main__':
    main()