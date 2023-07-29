

from typing import Tuple
from chebyshev import GridwiseChebyshev,ChebyshevInterval
from solver.lclsolve import LocalSysAllocator
from solver.eqgen import EquationFactory
from solver.linsolve import GlobalSystemSolver
import numpy as np

    
class LossFunction:
    def gradient(self,u:GridwiseChebyshev)->Tuple[np.ndarray,np.ndarray]:...
    

class Edges2Coeffs:
    def __init__(self,equfact:EquationFactory,dim:int):
        self.averaging_edge_value = equfact.bndr.averaging_edge_value
        self.dim = dim
    def product(self,ldeg:int,rdeg:int,dldedge:np.ndarray,output:np.ndarray,ind0:int,ind1:int):
        ls,rs = self.averaging_edge_value(ldeg,rdeg)
        slc0 = slice(ind0,ind0 + ldeg)
        slc1 = slice(ind1,ind1 + rdeg)
        output[slc0,:] += np.outer(ls,dldedge[0])
        output[slc1,:] += np.outer(rs,dldedge[1])      
    def transpose_edge_product(self,dldedge:np.ndarray,degrees:np.ndarray,):        
        # dldedge = dldedge.reshape([-1,self.dim])
        nedges = dldedge.shape[0]
        output = np.zeros((np.sum(degrees),self.dim),)
        assert len(degrees) == dldedge.shape[0] + 1
        cdeg = np.cumsum(degrees)
        cdeg = np.insert(cdeg,0,0)
        for i in range(nedges):
            self.product(degrees[i],degrees[i+1],dldedge[i],output,cdeg[i],cdeg[i+1])
        return output.flatten()
    
    

class AdjointMethod(Edges2Coeffs):
    def __init__(self,equac:EquationFactory,dim:int):
        Edges2Coeffs.__init__(self,equac,dim)
    def get_adjoint(self,state:GridwiseChebyshev,globalsys:GlobalSystemSolver,dloss_dstate:np.ndarray):
        preadjoint = self.transpose_edge_product(dloss_dstate,state.extended_ps,)
        adjglb = globalsys.adjoint_system()
        adjglb.rhs = preadjoint
        adjglb.solve()
        adjoint_weights = adjglb.solution
        
        admid = adjoint_weights[:-2*self.dim]
        adhead = adjoint_weights[-2*self.dim:-self.dim]
        adtail = adjoint_weights[-self.dim:]
        adjoint_weights = np.concatenate([adhead,admid,adtail])
        adjoint = state.create_from_solution(adjoint_weights,self.dim)  
        return adjoint
    
    
class DesignProduct(LocalSysAllocator):
    def __init__(self,lsa:LocalSysAllocator):
        self.__dict__.update(lsa.__dict__)
    def interval_dot(self,solution:ChebyshevInterval,adjoint:ChebyshevInterval,design_gcheb:ChebyshevInterval)->float:
        mat,rhs = self.create_quadrature_block(design_gcheb,solution.degree)
        gchebr = mat@solution.coeffs.flatten() - rhs
        return gchebr @ adjoint.coeffs.flatten()
    def dot(self,solution:GridwiseChebyshev,\
                       adjoint:GridwiseChebyshev,\
                        design_gcheb:GridwiseChebyshev):        
        gradient = []
        for sltn,adj,dgch in zip(solution,adjoint,design_gcheb):
            gradient.append(self.interval_dot(sltn,adj,dgch))
        return np.array(gradient)

        