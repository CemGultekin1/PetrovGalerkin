from chebyshev import ResidualNorm
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

'''


class OrthogonalResidual:
    def __init__(self,leftslv:LocalSystemSolver,rightslv:LocalSystemSolver) -> None:
        pass