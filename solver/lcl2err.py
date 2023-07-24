import logging
from .linsolve import LocalSystemSolver
import numpy as np
from chebyshev import NumericType,coeffevl,GridwiseChebyshev,ChebyshevInterval
import numpy.polynomial.chebyshev as cheb
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
    def __init__(self,dim:int,chebint:ChebyshevInterval,lclslv:LocalSystemSolver,) -> None:
        lclslv.solve()
        degree = chebint.degree
        gcheb = GridwiseChebyshev.create_from_local_solution(chebint,\
                        lclslv.interior_solution,\
                        lclslv.edge_solution,dim**2*degree)
        matfun,_ = chebint.separate_funs()
        self.dim = dim
        self.solution = gcheb
        self.matfun = matfun
        self.degree = degree
        self.h = chebint.h
        self.interval = chebint.interval
    def __call__(self,x:NumericType):
        mat = self.matfun(x)
        sltn = self.solution(x) # t x d x deg * d
        mat = mat.reshape([len(x),self.dim,self.dim]).transpose(0,2,1) # t x d x d
        sltn = sltn.reshape(len(x),self.dim,self.degree*self.dim) # t x d x deg*d
        # ATv = np.tensordot(mat,sltn,axes = (2,1)) # t x d x deg*d
        ATv = np.matmul(mat,sltn)
        # logging.info(f'ATv.shape = {ATv.shape}')
        ucoeff = np.eye(self.degree)
        a,b = self.interval
        xhat = (x - a)/(b-a)*2 - 1
        
        dv = cheb.chebder(self.solution.cheblist[0].coeffs)*2/self.h
        ucoeff[:-1,:] += dv
        dvpu :np.ndarray= coeffevl(xhat,ucoeff) # t x deg
        dvpu = dvpu.reshape([len(x),1,-1,1])*np.eye(self.dim).reshape([1,self.dim,1,self.dim])
        dvpu = dvpu.reshape([len(x),self.dim,self.degree*self.dim])        
        dv = coeffevl(x,dv).reshape([-1,self.dim,self.degree*self.dim]) # t x d x deg*d
        r = - dvpu + ATv 
        return r
        
        