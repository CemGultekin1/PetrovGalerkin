from solver.dimensional import Dimensional
from .linsolve import LocalSystemSolver
import numpy as np
from chebyshev import NumericType,coeffevl,GridwiseChebyshev,ChebyshevInterval,coeffgen,ResidualNorm
import numpy.polynomial.chebyshev as cheb
from .eqgen import EquationFactory
from .lclsolve import LocalSysAllocator
from chebyshev import GridwiseChebyshev,ChebyshevInterval
from .glbsys import DenseLocalSystem
from .linsolve import LocalSystemSolver


'''
Given v that solves
<U,L*v> = u
we get residue
e = L*v - u
we want L2 norm of its residual orthogonal to U 
||r||^2

How to compute square norm?
r has degree 3p
r^2 has degree 6p

*   Implement r as a function
    **  r(x) = -v'(x) -AT(x)v(x) -u(x)
        or
        r(x) = +v'(x) +AT(x)v(x) +u(x)
        
*   Implement residual norm computation
    **  Given a function,f(x), and degree,p,
        compute ||f^perp||^2
    
        *** Get 2p=q>p coefficients of f
            f = f_i u_i, i=1,..,q
            <f,u_j> = f_i <u_j,u_i>
            ||r||^2 = 
            ||f - fp||^2 = 
            (f - fp)@Q@(f - fp) 
            Qij = <u_i,u_j>
            
        *** Minimize this for fp
            E@Q@(f-fp) = 0
            (E@Q@E)@fp = E@Q@f
            fp = E.T@(E@Q@E)\(E@Q@f)
            
        *** Compute ||r||^2 using 
            (f - fp)@Q@(f - fp) 
            
        *** Larger expression
            A = E.T@inv(E@Q@E)
            B = A@E@Q  
            L = (I-B)^T@Q@(I-B)
            ||r||^2 = f@L@f

    **  We can store L matrices for all possible values of p


    (deg*dim x deg*dim) @ (deg*dim x deg*dim)  = (deg*dim x deg*dim)
'''

class ResidualFunction:
    def __init__(self,dim:int,chebint:ChebyshevInterval,probint:ChebyshevInterval,lclslv:LocalSystemSolver,) -> None:
        lclslv.solve()
        degree = chebint.degree
        gcheb = GridwiseChebyshev.create_from_local_solution(chebint,\
                        lclslv.interior_solution,\
                        lclslv.edge_solution,dim**2*degree)
        matfun,_ = probint.separate_funs()
        self.dim = dim
        self.solution = gcheb
        self.matfun = matfun
        self.degree = degree
        self.h = chebint.h
        self.interval = chebint.interval
    def __call__(self,x:NumericType):
        mat :np.ndarray= self.matfun(x)
        sltn:np.ndarray= self.solution(x) # t x d x deg * d
        mat = mat.reshape(len(x),self.dim,self.dim).transpose(0,2,1) # t x d x d
        sltn = sltn.reshape(len(x),self.dim,self.dim) # t x d
        negATv = np.matmul(mat,sltn)
        a,b = self.interval
        xhat = (x - a)/(b-a)*2 - 1        
        dv = cheb.chebder(self.solution.cheblist[0].coeffs)*2/self.h
        dv :np.ndarray= coeffevl(xhat,dv) # t x dim
        dv = dv.reshape(len(x),self.dim,self.dim)
        r = - dv + negATv 
        return r
                
class OrthogonalResidueNorm:
    def __init__(self,maxdegree:int,) -> None:
        self.degree = maxdegree
        self.resnorm = ResidualNorm(self.degree)
        self.resnorm.fillup()
    def orthogonal_norm(self,res:ResidualFunction,degree:int):
        return self.resnorm.residual_norm_from_fun(res,res.interval,degree = degree)
        
class LocalErrorEstimate(Dimensional):
    def __init__(self,equfact:EquationFactory) -> None:
        self.lcl_sys_alloc = LocalSysAllocator(equfact)
        self.orth_res_norm = OrthogonalResidueNorm(equfact.max_degree)
    def interval_error(self,lclcheb:ChebyshevInterval,problem_components:ChebyshevInterval):
        blocks,rhs = self.lcl_sys_alloc.get_single_interval_blocks(lclcheb,problem_components)
        sgs = DenseLocalSystem(blocks,rhs)
        lss = LocalSystemSolver(sgs)
        lss.solve()
        res = ResidualFunction(self.dim,lclcheb,problem_components,lss)
        orthnorm = self.orth_res_norm.orthogonal_norm(res,lclcheb.degree)
        return orthnorm*lclcheb.h/2



