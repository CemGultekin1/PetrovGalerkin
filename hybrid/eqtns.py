from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Tuple
import numpy as np
from hybrid.handles import DesignParameteric, Parameteric
from hybrid.symeq import org_sys,params_sys,design_sys,vnames
from chebyshev import GridwiseChebyshev

def non_scalar_input_wrapper(fun):
    def non_scalar_wrapped_fun(self,*args, **kwargs):
        if not np.isscalar(args[0]):
            jac = list(map(lambda args_: fun(self,*args_,**kwargs),list(zip(*args))))
            return np.stack(jac,axis = 0)
        return fun(self,*args,**kwargs)
    return non_scalar_wrapped_fun
def boundary_wrapper(fun):
    def boundary_wrapped_fun(self,*args,**kwargs):
        b1 = fun(self,*args,**kwargs)
        b0 = np.eye(b1.shape[0])
        rhs = np.zeros((b0.shape[0],))
        return b0,b1,rhs
    return boundary_wrapped_fun

default_params_dict = dict(
    m0s = 0.1,
    invt1 = 1/1.6,
    invt2f = 1/.065,
    r = 30,    
    t2s = 60e-6
)

class HybridSystemEquations(Parameteric):
    params_list :List[str] = list(default_params_dict.keys())    
    _params:Tuple[float,...]
    def __init__(self,**kwargs) -> None:
        kwargs1 = deepcopy(default_params_dict)
        kwargs1.update(kwargs)
        self._params = ()
        self.__dict__.update(kwargs1)
    def change_of_parameters(self, **kwargs):
        self._params = ()
        return super().change_of_parameters(**kwargs)
    @property
    def params_dict(self,):
        return dict(zip(self.solved_param_names,self.params))
    @property
    def params(self,):
        if not bool(self._params):
            args = tuple(
                self.__dict__[key] for key in vnames if key in self.__dict__
            )
            self._params = args
        return self._params
    @property
    def param_names(self,):
        vn = [key for key in vnames if key in self.__dict__]
        return ['m'] + list(vn)
    @property
    def solved_param_names(self,):
        vn = [key for key in vnames if key in self.__dict__]
        return list(vn)
    @property
    def param_values_str(self,):
        prmstrs = ["{:.1e}".format(prm) for prm in self.params]
        return [f'{vn} = {prmstr}' for vn,prmstr in zip(vnames,prmstrs)]
    @property
    def signal_names(self,):
        return ['$r^f$','$z^s$']
    @non_scalar_input_wrapper
    def params_sys_mat(self,*args,):
        return params_sys(*self.params,*args,mat_flag=True)
    @non_scalar_input_wrapper
    def params_sys_rhs(self,*args,):
        return params_sys(*self.params,*args,rhs_flag=True)    
    
    @non_scalar_input_wrapper
    def org_sys_mat(self,*args,):
        return org_sys(*self.params,*args,mat_flag=True)
    @non_scalar_input_wrapper
    def org_sys_rhs(self,*args,):
        return org_sys(*self.params,*args,rhs_flag=True)
    
    @non_scalar_input_wrapper
    def design_sys_mat(self,*args,name:str = 'theta1'):
        return design_sys(*self.params,*args,mat_flag=True,name = name)
    @non_scalar_input_wrapper
    def design_sys_rhs(self,*args,name:str = 'theta1'):
        return design_sys(*self.params,*args,rhs_flag=True,name = name)


    @boundary_wrapper
    def params_sys_bndr(self,*args,):
        return params_sys(*self.params,*args,bndr_flag=True)
    @boundary_wrapper
    def org_sys_bndr(self,*args,):
        return org_sys(*self.params,*args,bndr_flag=True)
    @boundary_wrapper
    def design_sys_bndr(self,*args,name:str = 'theta1'):
        return design_sys(*self.params,*args,bndr_flag=True,name = name)
    
    
class DesignFunctions(DesignParameteric):
    tr:float = 3.5e-3
    params_list :List[str] = []
    design_params_list : List[str]  = 'theta_seq trf_seq'.split()
    def __init__(self,theta_seq:np.ndarray,trf_seq:np.ndarray) -> None:
        self.theta_seq = theta_seq
        self.trf_seq = trf_seq
        assert len(trf_seq) == len(theta_seq) - 1
    @property
    def total_time(self,):
        return self.tr*len(self.theta_seq)
    @property
    def num_fp(self,):
        return len(self.theta_seq)
    def give_theta_interval(self,i:int,):
        if i==0:
            x11 = self.tr*(i+1) + self.trf_seq[i]/2
            x00 = 0
            return (),(x00,x11)
        elif i == self.num_fp - 1:
            x11 = self.tr*(i+1)
            x01 = self.tr*i + self.trf_seq[i-1]/2
            x00 = self.tr*i - self.trf_seq[i-1]/2
            return (x00,x01), (x01,x11)
        else:
            x11 = self.tr*(i+1) + self.trf_seq[i]/2
            # x10 = self.tr*(i+1) - self.trf_seq[i+1]/2
            x01 = self.tr*i + self.trf_seq[i-1]/2
            x00 = self.tr*i - self.trf_seq[i-1]/2
            return (x00,x01),(x01,x11)
    def design_gradient_collection(self,dldth1:np.ndarray,dldth2:np.ndarray,dldf_dfdth:np.ndarray,solution:GridwiseChebyshev):
        for i in range(dldf_dfdth.shape[0]):
            theta2int,theta1int = self.give_theta_interval(i)
            if bool(theta2int):
                tchints2 = solution.find_touching_intervals(*theta2int)
                dldf_dfdth[i] += np.sum(dldth2[tchints2])
            if bool(theta1int):
                tchints1 = solution.find_touching_intervals(*theta1int)
                dldf_dfdth[i] += np.sum(dldth1[tchints1])
        return dldf_dfdth
    def __call__(self,t):
        if not np.isscalar(t):
            jac = list(map(self.__call__,t))
            return np.stack(jac,axis = 1)
        t = t% self.total_time
        clsst = np.round(t/self.tr).astype(int)
        # theta1,theta2        
        # reltime,freq,rfpulse      
        if clsst == self.num_fp:
            return np.array([self.theta_seq[-1],self.theta_seq[-1],0,0,0])
        elif clsst == 0:
            return np.array([self.theta_seq[0],self.theta_seq[0],0,0,0])
        
        trf = self.trf_seq[clsst-1]
        dst = t - clsst*self.tr
        dst = dst/(trf/2)
        th0 = self.theta_seq[clsst-1]
        th1 = self.theta_seq[clsst]
        dst = (dst+1)/2
        if dst < 0:
            return np.array([th0,th0,0,0,0])
        elif dst > 1:
            return np.array([th1,th1,0,0,0])
            
        reltime = dst*trf                
        
        return np.array([th0,th1,reltime,1/trf,1])
    
class TimeDependentHSS(HybridSystemEquations,DesignFunctions):
    def __init__(self,theta_fun:np.ndarray,trf_fun:np.ndarray,**kwargs):
        HybridSystemEquations.__init__(self,**kwargs)        
        DesignFunctions.__init__(self,theta_fun,trf_fun)
        
    def __getattribute__(self, __name: str) -> Any:
        if 'sys' in __name:
            fun = super().__getattribute__(__name)
            # if 'bndr' not in __name:
            def wrapped_fun(t,**kwargs):
                args = self(t)
                return fun(*args,**kwargs)
            # else:
            #     args = self(0)
            #     def wrapped_fun(**kwargs):
            #         return fun(*args,**kwargs)
            return wrapped_fun                
        return super().__getattribute__(__name)
    
    
class DesignSequences(DesignFunctions):
    def __init__(self, n:int = 2,random_initialization_seed :int = -1,theta_seq:np.ndarray = None,trf_seq:np.ndarray = None) -> None:
        if theta_seq is None:
            if random_initialization_seed >= 0:
                np.random.seed(random_initialization_seed)
                theta_seq = np.random.rand(n)*np.pi/2
            else:
                theta_seq = np.ones((n,))*np.pi/2
        else:
            theta_seq = theta_seq
        if trf_seq is None:
            trf_seq = np.ones((n-1,))*5e-4
        else:
            trf_seq = trf_seq
            assert len(trf_seq) == len(theta_seq) - 1
        super().__init__(theta_seq, trf_seq)