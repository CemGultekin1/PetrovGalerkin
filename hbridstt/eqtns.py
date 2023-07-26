from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
from chebyshev   import NumericType
from hbridstt.symeq import org_sys,params_sys,design_sys,vnames
def non_scalar_input_wrapper(fun):
    def wrapped_fun(self,theta,dtheta, **kwargs):
        if not np.isscalar(theta):
            jac = list(map(lambda x,y: fun(self,x,y,**kwargs),theta,dtheta))
            return np.stack(jac,axis = 0)
        return fun(self,theta,dtheta,**kwargs)
    return wrapped_fun
def boundary_wrapper(fun):
    def wrapped_fun(self,theta,dtheta,**kwargs):
        b1 = fun(self,theta,dtheta,**kwargs)
        b0 = np.eye(b1.shape[0])
        rhs = np.zeros((b0.shape[0],))
        return b0,b1,rhs
    return wrapped_fun
@dataclass
class HybridEq:
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
    def params_sys_mat(self,theta,dtheta,):
        return params_sys(*self.params,theta,dtheta,mat_flag=True)
    @non_scalar_input_wrapper
    def params_sys_rhs(self,theta,dtheta,):
        return params_sys(*self.params,theta,dtheta,rhs_flag=True)    
    
    @non_scalar_input_wrapper
    def org_sys_mat(self,theta,dtheta,):
        return org_sys(*self.params,theta,dtheta,mat_flag=True)
    @non_scalar_input_wrapper
    def org_sys_rhs(self,theta,dtheta,):
        return org_sys(*self.params,theta,dtheta,rhs_flag=True)
    
    @non_scalar_input_wrapper
    def design_sys_mat(self,theta,dtheta,name:str = 'theta'):
        return design_sys(*self.params,theta,dtheta,mat_flag=True,name = name)
    @non_scalar_input_wrapper
    def design_sys_rhs(self,theta,dtheta,name:str = 'theta'):
        return design_sys(*self.params,theta,dtheta,rhs_flag=True,name = name)


    @boundary_wrapper
    def params_sys_bndr(self,theta,dtheta,):
        return params_sys(*self.params,theta,dtheta,bndr_flag=True)
    @boundary_wrapper
    def org_sys_bndr(self,theta,dtheta,):
        return org_sys(*self.params,theta,dtheta,bndr_flag=True)
    @boundary_wrapper
    def design_sys_bndr(self,theta,dtheta,name:str = 'theta'):
        return design_sys(*self.params,theta,dtheta,bndr_flag=True,name = name)
    
    
class ThetaFun:
    tr:float = 3.5e-3
    def __init__(self,theta_seq,trf_seq) -> None:
        self.theta_seq = theta_seq
        self.trf_seq = trf_seq
        assert len(trf_seq) == len(theta_seq) - 1
        self.total_time = self.tr*len(theta_seq)
        self.num_int = len(theta_seq)
    def __call__(self,t):
        if not np.isscalar(t):
            jac = list(map(self.__call__,t))
            return np.stack(jac,axis = 1)
        t = t% self.total_time
        clsst = np.round(t/self.tr).astype(int)
        if clsst == self.num_int:
            return np.array([self.theta_seq[-1],0])
        elif clsst == 0:
            return np.array([self.theta_seq[0],0])
        
        trf = self.trf_seq[clsst-1]
        dst = t - clsst*self.tr
        dst = dst/(trf/2)
        dst = (dst+1)/2
        dur = np.maximum(np.minimum(dst,1),0)
        
        th0 = self.theta_seq[clsst-1]
        th1 = self.theta_seq[clsst]
        th = (th1 - th0)*dur + th0
        return np.array([th,(th1 - th0)/trf])

class HybridStateSystem(HybridEq,ThetaFun):
    def __init__(self,theta_fun:np.ndarray,trf_fun:np.ndarray,**kwargs):
        HybridEq.__init__(self,**kwargs)
        ThetaFun.__init__(self,theta_fun,trf_fun)
    def __getattribute__(self, __name: str) -> Any:
        if 'sys' in __name:
            fun = super().__getattribute__(__name)
            if 'bndr' not in __name:
                def wrapped_fun(t,**kwargs):
                    th,dth = self(t)
                    return fun(th,dth,**kwargs)
            else:
                def wrapped_fun(**kwargs):
                    return fun(0,0,**kwargs)
            return wrapped_fun                
        return super().__getattribute__(__name)
    
def main():
    import matplotlib.pyplot as plt
    np.random.seed(0)
    thetaseq = np.random.rand(5,)
    trfseq = np.random.rand(4)*1e-3
    hss = HybridStateSystem(thetaseq,trfseq)
    b0,b1,rhs = hss.design_sys_bndr(name = 'theta')
    print(b0.shape,b1.shape,rhs.shape)
    
if __name__ == '__main__':
    main()