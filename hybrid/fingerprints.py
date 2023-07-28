from typing import Any
from hybrid.eqtns import TimeDependentHSS
import numpy as np
from chebyshev import GridwiseChebyshev



class HybridStateSystem(TimeDependentHSS):
    def __init__(self,num:int = -1,time:float = -1.,mode:str = 'org',design_param :str = 'theta1',theta:np.ndarray = None) -> None:
        if theta is None:            
            if time > 0:
                n = np.floor(time/self.tr).astype(int)
            elif num > 0:
                n = num
            else:
                raise Exception
            theta = np.random.rand(n)*np.pi/4 + np.pi/4
        else:
            n = len(theta)
            
        self.design_param = design_param
        self.mode = mode
        np.random.seed(0)
        
        trfs = np.ones((n-1))*5e-4
        # alphas = (-1)**np.arange(n)*alphas
        super().__init__(theta,trfs)
        self.dim = 2
    @property
    def signal_names(self,):
        snms = super().signal_names
        if self.mode == 'org':
            return snms
        prms = super().param_names
        return [f'd{snm}/d{prm}' for snm in snms for prm in prms]
    def edges(self,full:bool = False):
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
        
    def fingerprint_edges(self,):
        return self.tr*np.arange(1,self.num_fp)
        
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


class HybridStateSolution(GridwiseChebyshev,HybridStateSystem):
    def __init__(self, gcheb:GridwiseChebyshev,theta_fun:np.ndarray,trf_fun:np.ndarray,**kwargs) -> None:
        HybridStateSystem.__init__(self,theta_fun,trf_fun,**kwargs)
        self.__dict__.update(gcheb.__dict__)
        
class Fingerprints:
    def __init__(self,hss:HybridStateSolution) -> None:
        
        self.fingerprint_times = np.arange(1,hss.num_fp)*hss.tr
        self.theta_seq = hss.theta_seq
        
        fingerprint_edges = hss.find_closest_edges(self.fingerprint_times)
        self.edges = fingerprint_edges
        avgvals = hss.edge_values.values.mean(axis = 1)        
        avg_theta = (hss.theta_seq[1:]+ hss.theta_seq[:-1])/2
        self.avg_state_vals = avgvals
        self.sine_weights = np.sin(avg_theta).reshape([-1,1])
        self.states :np.ndarray= avgvals[fingerprint_edges,::2]
        self.values:np.ndarray = self.states*self.sine_weights
        
    def state_avg_edges_derivative_inner_product(self,dldf:np.ndarray):
        '''
        given dldf, returns dldu_avg
        f = u*sin
        dldu = dldf @ dfdu = dldf * sin
        '''
        dldu_avg = np.zeros(self.avg_state_vals.shape)
        dldu_avg[self.edges,::2] = dldf*self.sine_weights
        return dldu_avg
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