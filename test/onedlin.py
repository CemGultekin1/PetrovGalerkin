
import logging
import numpy as np
from chebyshev import coeffgen,coeffevl
import numpy.polynomial.chebyshev as cheb
from scipy.linalg import expm
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

class RandomLinearODE:
    def __init__(self,dim:int,degree:int = 3,seed:int = 0) -> None:
        self.dim = dim
        self.degree = degree
        np.random.seed(seed)
        weights = np.arange(degree)**5
        weights = weights/weights.sum()
        weights = weights.reshape(-1,1)
        
        self.mat_coeffs = np.random.randn(degree,dim**2)*weights
        self.rhs_coeffs = np.random.randn(degree,dim)*weights
        self.bounds = (0,1)
        self.initial_point = np.random.randn(dim,)*0
        self.solution_coeffs = None
    def matfun(self,x):
        x = x*2 -1
        if np.isscalar(x):
            return self.matfun(np.array([x]))
        return coeffevl(x,self.mat_coeffs).reshape([len(x),self.dim,self.dim,-1]).sum(axis = 3)
    def rhsfun(self,x):
        x = x*2 -1
        if np.isscalar(x):
            return self.rhsfun(np.array([x]))
        return coeffevl(x,self.rhs_coeffs).reshape([len(x),self.dim])
    def integrate(self,):
        int_mat_coeffs = cheb.chebint(self.mat_coeffs)/2
        def exponential_mat_product(x):            
            def singular_fun(x_):
                ymat = coeffevl(x_,-int_mat_coeffs) - coeffevl(0,-int_mat_coeffs)
                ymat = ymat.reshape(self.dim,self.dim,)
                rhs =  coeffevl(x_,self.rhs_coeffs)
                rhs = rhs.reshape(self.dim,)
                y =expm(ymat)@rhs
                return y.reshape(self.dim)
            return np.stack(list(map(singular_fun,x)),axis = 0)
        expmrhsprod = coeffgen(exponential_mat_product,self.degree*8,outbounds = self.bounds)
        expmrhsprod = expmrhsprod.reshape(-1,self.dim,)
        yp = cheb.chebint(expmrhsprod)
        def solution_fun(x_):
            def singular_fun(x):
                rhs = coeffevl(x,yp) - coeffevl(0,yp)
                mat = coeffevl(x,int_mat_coeffs) - coeffevl(0,int_mat_coeffs)
                mat = mat.reshape(self.dim,self.dim,-1).sum(axis = 2)
                y =expm(mat)@(rhs + self.initial_point)
                return y.reshape(self.dim)
            return np.stack(list(map(singular_fun,x_)),axis = 0)
        self.solution_coeffs = coeffgen(solution_fun,self.degree*8,outbounds = self.bounds)
        # logging.info(f'self.solution_coeffs.shape = {self.solution_coeffs.shape}')
        assert self.solution_coeffs.shape == (self.degree*8 + 1, self.dim)
    def eval_solution(self,x):
        x = x*2 -1
        return coeffevl(x,self.solution_coeffs)
        
        
        
def main():
    rlode = RandomLinearODE(1,degree = 3)
    rlode.integrate()
    x = np.linspace(0,1,256)
    y = rlode.eval_solution(x)
    plt.plot(x,y)
    plt.savefig('dummy.png')
    

if __name__ == '__main__':
    main()