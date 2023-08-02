from typing import Tuple
from chebyshev import GridwiseChebyshev,ChebyshevInterval
from solver.lclsolve import LocalSysAllocator
from solver.eqgen import EquationFactory
from solver.linsolve import GlobalSystemSolver
import numpy as np

    

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


class TimeInstanceDesignProduct(DesignProduct):
    def interval_dot(self,solution:Tuple[ChebyshevInterval,ChebyshevInterval],\
                        adjoint:Tuple[ChebyshevInterval,ChebyshevInterval],\
                        problem:Tuple[ChebyshevInterval,ChebyshevInterval])->float:
        left_right = self.time_derivative(solution,problem)
        outs = 0.
        for (mat,rhs),sol,adj in zip(left_right,solution,adjoint):
            gchebr = mat@sol.coeffs.flatten() - rhs
            outs += gchebr @ adj.coeffs.flatten()
        return outs
    def dot(self,solution:GridwiseChebyshev,\
                       adjoint:GridwiseChebyshev,\
                        design_gcheb:GridwiseChebyshev):        
        gradient = []
        def one_off_zipper(*gcheb:GridwiseChebyshev):
            prl = []
            for gce in gcheb:
                cheb = gce.cheblist
                prl.append(zip(cheb[:-1],cheb[1:]))
            return zip(*prl)
        for sltn,adj,dgch in one_off_zipper(solution,adjoint,design_gcheb):
            gradient.append(self.interval_dot(sltn,adj,dgch))
        return np.array(gradient)
    def time_derivative(self,
        solutins:Tuple[ChebyshevInterval,ChebyshevInterval],\
        problems:Tuple[ChebyshevInterval,ChebyshevInterval])\
            -> Tuple[Tuple[np.ndarray,np.ndarray],Tuple[np.ndarray,np.ndarray]]:
        '''
        
        g2(t2) - g1(t1) + int_{t1}^{t2} f( xhat(x,t1,t2) ) dx = I(t1,t2,f)
        
        xhat(x,t1,t2) = (x  -  (t2 + t1)/2)/(t2 -t1)*2
        xhat(x,t1,t2,td) = (x  -  (t2 + t1)/2)/dt*2
        
        d xhat /d t1 = - 1/dt 
        d xhat /d dt = - xhat/dt
        d xhat /d t2 = - 1/dt
        
        d xhat /d t1 = - 1/dt + xhat /dt = -(1 - xhat)/dt
        d xhat /d t2 = - 1/dt - xhat /dt = -(1 + xhat)/dt
        
        dIdt1 = -f(0) - I(t1,t2,T1*f')
        dIdt2 =  f(1) - I(t1,t2,T1*f')/(t2-t1)*1/(t2- t1)
        generate_quad_interior_time_derivatives_element
        '''
        elems = []
        for prob,sol,left in zip(problems,solutins,[False,True]):
            bndr_tder = self.local_equation.generate_quad_boundary_time_derivatives_element(
                prob,sol.degree,left = left)
            int_tder = self.local_equation.generate_quad_interior_time_derivatives_element(
                prob,sol.degree,left = left)
            mats,rhss = zip(bndr_tder,int_tder)
            elems.append((np.add(*mats),np.add(*rhss)))
        return tuple(elems)

        
        
        
def main():
    from chebyshev import ListOfFuns
    t = 1
    targdeg = 2
    probdeg = 5
    
    dim = 2
    flof = ListOfFuns(lambda x: np.stack([x,x**2,x**3,x**4],axis = 1).reshape([len(x),2,2]),\
                            lambda x:np.stack([np.cos(x),x**2],axis = 1).reshape([len(x),2])\
                                ).flatten()
    chebint = ChebyshevInterval.from_function(flof,probdeg,0,t,)
    chebint2 = ChebyshevInterval.from_function(flof,probdeg,t,2,)
    
    solchebint = ChebyshevInterval.from_function(flof,targdeg,0,t,)
    solchebint2 = ChebyshevInterval.from_function(flof,targdeg,t,2,)
    eqf = EquationFactory(8,)
    lsa = LocalSysAllocator(eqf)    
    
    tidp = TimeInstanceDesignProduct(lsa)
    tidp.set_dim(dim)
    
    dleftdt,drightdt = tidp.time_derivative((solchebint,solchebint2,),(chebint,chebint2))
    
    def eval(t):
        chebint = ChebyshevInterval.from_function(flof,probdeg,0,t,)
        chebint2 = ChebyshevInterval.from_function(flof,probdeg,t,2,)
        left = tidp.local_equation.generate_interior_matrices(chebint,1,targdeg,targdeg)
        left.trim(right = True,left = True)
        right = tidp.local_equation.generate_interior_matrices(chebint2,targdeg,1,targdeg)
        right.trim(right = True,left = True)
        return np.stack([left.rhsclm,right.rhsclm],axis = 1)
    
    
    g = np.stack([dleftdt[1],drightdt[1]],axis = 1)
    
    
    h = 1e-5
    x0 = eval(t )    
    x1 = eval(t+h)    
    dxh = (x1 - x0)/h
    relerr = np.linalg.norm(dxh - g)/(np.linalg.norm(dxh) + np.linalg.norm(g))*2
    print(f'relerr = {relerr}')
    
if __name__ == '__main__':
    main()