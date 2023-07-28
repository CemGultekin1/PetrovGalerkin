

from typing import List, Tuple
from chebyshev import GridwiseChebyshev,ChebyshevInterval
from solver.lclsolve import LocalSysAllocator
from solver.eqgen import EquationFactory
from solver.linsolve import GlobalSystemSolver
import numpy as np

class DesignSupport:
    evaluation_pts : List[float]
    evaluation_edges : List[int]
    def __init__(self) -> None:
        pass
    def __len__(self,)->int:...
    def get_design_span(self,design_ind:int,)->Tuple[float,float]:...
    def mean_value_matmul(self,):...
    
class LossFunction:
    dim:int
    def gradient(self,u:GridwiseChebyshev)->Tuple[np.ndarray,np.ndarray]:...
    

class AdjointSolver:
    def __init__(self,equfact:EquationFactory,lossfun:LossFunction,dim:int):
        self.equfact = equfact
        self.lossfun = lossfun
        self.dim = dim
    def averaging_edge_vec(self,ldeg:int,rdeg:int,):
        leftside,rightside = self.equfact.bndr.averaging_edge_value(ldeg,rdeg)
        return leftside,rightside
    def product(self,ldeg:int,rdeg:int,dldavg:np.ndarray,output:np.ndarray,ind0:int,ind1:int):
        ls,rs = self.averaging_edge_vec(ldeg,rdeg)
        slc0 = slice(ind0,ind0 + len(ls))
        slc1 = slice(ind1,ind1 + len(rs))
        output[slc0,:] += np.outer(ls,dldavg)
        output[slc1,:] += np.outer(rs,dldavg)        
    def get_adjoint(self,state:GridwiseChebyshev,globalsys:GlobalSystemSolver):
        dldstate,dldesign = self.lossfun.gradient(state)
        preadjoint = self.avg_transpose(dldstate,state.extended_ps,)
        adjglb = globalsys.adjoint_system()
        adjglb.rhs = preadjoint
        adjglb.solve()      
        adjoint_weights = adjglb.solution
        adjoint = state.create_from_solution(adjoint_weights,self.dim)  
        return adjoint,dldesign
    def avg_transpose(self,dldavg:np.ndarray,degrees:np.ndarray,):        
        dldavg = dldavg.reshape([-1,self.dim])
        nedges = dldavg.shape[0]
        output = np.zeros((np.sum(degrees),self.dim),)
        for i in range(nedges):
            self.product(degrees[i],degrees[i+1],dldavg[i],output,i,i+1)
        return output.flatten()
    
    
class IntervalDesignProduct:
    def __init__(self,lsa:LocalSysAllocator):
        self.local_sys_allocator = lsa
    def __call__(self,solution:ChebyshevInterval,adjoint:ChebyshevInterval,design_gcheb:ChebyshevInterval)->float:
        bmf,_ = self.local_sys_allocator.get_single_interval_blocks(solution, design_gcheb, without_boundary=True)
        mat = bmf.mat_blocks[0].matblock
        rhs = bmf.rhs_blocks[0].matblock
        gchebr = -mat@solution.coeffs.flatten() + rhs
        return gchebr @ adjoint.coeffs.flatten()

    
class GlobalDesignProduct(IntervalDesignProduct):
    def __init__(self,des_sup:DesignSupport,lsa:LocalSysAllocator) -> None:
        super().__init__(lsa)
        self.support = des_sup        
    def __call__(self,solution:GridwiseChebyshev,\
                       adjoint:GridwiseChebyshev,\
                        design_gcheb:GridwiseChebyshev):
        gradient = []
        for i in range(len(self.support)):
            x0,x1 = self.support.get_design_span(i)
            intervals = design_gcheb.find_touching_intervals(x0,x1)
            fl =  0
            for intv in intervals:
                sltn,adj,dgch = solution[intv],adjoint[intv],design_gcheb[intv]
                fl += super().__call__(sltn,adj,dgch)
            gradient.append(fl)
        return np.array(gradient)
            
class GradientFactory:
    def __init__(self,equac:EquationFactory,\
                lossfn:LossFunction,localsys:LocalSysAllocator,\
                    dessup:DesignSupport,dim:int) -> None:
        self.design_product = GlobalDesignProduct(dessup,localsys)
        self.adjoint_solver = AdjointSolver(equac, lossfn,dim)
        self.gradient = np.empty(0)
    def get_gradient(self,solution:GridwiseChebyshev,design_gcheb:GridwiseChebyshev,glbsys:GlobalSystemSolver):
        adjoint,dldesign = self.adjoint_solver.get_adjoint(solution,glbsys)
        grad =  self.design_product(solution,adjoint,design_gcheb,)
        return grad,dldesign
        
        