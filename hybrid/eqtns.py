from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
from hybrid.symeq import org_sys,params_sys,design_sys,vnames


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
@dataclass
class HybridSystemEquations:
    m0s:float = 0.1
    invt1:float = 1/1.6
    invt2f:float = 1/.065
    r:float = 30    
    t2s:float = 60e-6
    _params:Tuple[float,...] = ()
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
    
    
class ThetaFun:
    tr:float = 3.5e-3
    def __init__(self,theta_seq:np.ndarray,trf_seq:np.ndarray) -> None:
        self.theta_seq = theta_seq
        self.trf_seq = trf_seq
        assert len(trf_seq) == len(theta_seq) - 1
        self.total_time = self.tr*len(theta_seq)
        self.num_fp = len(theta_seq)
        self.last_t = None
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

class TimeDependentHSS(HybridSystemEquations,ThetaFun):
    def __init__(self,theta_fun:np.ndarray,trf_fun:np.ndarray,**kwargs):
        HybridSystemEquations.__init__(self,**kwargs)
        ThetaFun.__init__(self,theta_fun,trf_fun)
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