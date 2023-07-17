import itertools
from typing import Tuple, Union
import numpy as np
from chebyshev.core import LinearBoundaryProblem,BoundaryCondition,PetrovGalerkinSolverSettings,LinearSolver
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev
import matplotlib.pyplot as plt
def matfun(x:NumericType):
    return np.stack([x**i for i in range(9)],axis = 1).reshape([-1,3,3])
def rhsfun(x:NumericType):
    return np.stack([x**i for i in range(3)],axis = 1).reshape([-1,3])
def main():
    np.random.seed(0)
    d = 2
    lof = ListOfFuns(matfun,rhsfun)
    flof = lof.flatten()

    boundary_condition = (np.random.randn(d,d),np.random.randn(d,d))
    lbp = LinearBoundaryProblem((matfun,rhsfun),boundary_condition=boundary_condition)
    pgsb = PetrovGalerkinSolverSettings()
    ls = LinearSolver(pgsb,lbp)
    
    # degree = 4
    # gcheb = GridwiseChebyshev.from_function(flof,degree,0,1)
    
    # def plot(tag:str,n:int):
    #     x = np.linspace(0,1,n)
    #     y = gcheb(x)
    #     plt.plot(x,y)
    #     plt.plot(x,y)
    #     plt.savefig(f'dummy-{tag}.png')
    #     plt.close()
    # for i in range(6):
    #     plot(i,64)
    #     gcheb.refine(0)
    
    return
    
if __name__== '__main__':
    main()