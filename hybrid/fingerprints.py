from typing import Any
from hybrid.eqtns import TimeDependentHSS,DesignFunctions
import numpy as np
from chebyshev import GridwiseChebyshev
from solver.linsolve import GlobalSystemSolver


class HybridStateSystem(TimeDependentHSS):
    def __init__(self,driving_functions:DesignFunctions = DesignFunctions( np.ones((2),)*np.pi/2,np.ones((1,))*5e-4),\
                mode:str = 'org',design_param :str = 'theta1',**kwargs) -> None:
        self.design_param = design_param
        self.mode = mode
        super().__init__(driving_functions.theta_seq,driving_functions.trf_seq,**kwargs)
    @property
    def dim(self,):
        return len(self.signal_names)
    @property
    def signal_names(self,):
        snms = super().signal_names
        if self.mode == 'org':
            return snms
        prms = super().param_names
        return [f'd{snm}/d{prm}' for snm in snms for prm in prms]
    @property
    def starting_edges(self,):
        full = True
        if full:
            x0 = self.tr*np.arange(1,len(self.trf_seq)+1) - self.trf_seq/2
            x1 = self.tr*np.arange(1,len(self.trf_seq)+1) + self.trf_seq/2
            x = self.tr*np.arange(self.num_fp+1)
            edges = np.concatenate([x0,x1,x])
            edges.sort()        
            return tuple(edges.tolist())
        else:
            x = self.tr*np.arange(self.num_fp+1)       
            return tuple(x.tolist())
                    
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
            return self.org_sys_bndr(0,)
        elif self.mode == 'params':
            return self.params_sys_bndr(0,)
        elif self.mode == 'design':
            return  self.design_sys_bndr(0,name = self.design_param)


class HybridStateSolution(GridwiseChebyshev,):
    def __init__(self, gcheb:GridwiseChebyshev,theta_fun:np.ndarray,trf_fun:np.ndarray,gss:GlobalSystemSolver,hss:HybridStateSystem) -> None:
        self.__dict__.update(gcheb.__dict__)
        self.global_sys_sol = gss
        self.theta_seq = theta_fun
        self.trf_seq = trf_fun
        self.params_dict = hss.params_dict
        self.signal_names = hss.signal_names
        self.num_fp = len(self.theta_seq)
        self.tr = hss.tr
    
class Fingerprints:
    def __init__(self,hss:HybridStateSolution) -> None:       
        self.num_fp = hss.num_fp 
        self.tr =hss.tr
        self._signal_names = hss.signal_names[::2]
        self.fingerprint_times = self.fingerprint_edges()
        self.theta_seq = hss.theta_seq       
        self.params_dict = hss.params_dict 
        self.dim = hss.dim//2
        edges = hss.find_closest_edges(self.fingerprint_times)
        self.edges = edges
        self.edge_values = hss.edge_values.values
        
        avg_theta = (self.theta_seq[1:]+ self.theta_seq[:-1])/2
        self.avg_state_vals = self.edge_values.mean(axis = 1)        
        self.sine_weights = np.sin(avg_theta).reshape([-1,1])
        self.states :np.ndarray= self.avg_state_vals[self.edges,::2]
        self.values = self.states*self.sine_weights
    @property
    def signal_names(self,):
        if len(self._signal_names) == 1:
            return self._signal_names[0]
        else:
            return self._signal_names
    def fingerprint_edges(self,):
        return self.tr*np.arange(1,self.num_fp)
        
    def update(self,):
        avg_theta = (self.theta_seq[1:]+ self.theta_seq[:-1])/2
        self.avg_state_vals = self.edge_values.mean(axis = 1)        
        self.sine_weights = np.sin(avg_theta).reshape([-1,1])
        self.states :np.ndarray= self.avg_state_vals[self.edges,::2]
        self.values = self.states*self.sine_weights
        
    def state_avg_edges_derivative_inner_product(self,dldf:np.ndarray):
        '''
        given dldf, returns dldu_avg
        f = u*sin
        dldu_avg = dldf @ dfdu_avg = dldf * sin
        '''
        dldu_avg = np.zeros(self.avg_state_vals.shape)
        dldu_avg[self.edges,::2] = dldf*self.sine_weights
        return dldu_avg
    def state_edges_derivative_inner_product(self,dldf:np.ndarray):
        dldu_avg = self.state_avg_edges_derivative_inner_product(dldf)
        zmat = np.zeros(self.edge_values.shape)
        zmat[:,0,:] = dldu_avg
        zmat[:,1,:] = dldu_avg
        return zmat
    def design_derivative_inner_product(self,dldf:np.ndarray):
        '''
        given dldf returns dldtheta
        '''
        dldavgtheta = dldf*self.states*np.sqrt(1 - self.sine_weights**2)
        dldavgtheta  = np.sum(dldavgtheta,axis = 1)
        n = len(self.theta_seq)
        dldtheta = np.zeros(n)
        dldtheta[1:] += dldavgtheta/2
        dldtheta[:-1] += dldavgtheta/2
        return dldtheta