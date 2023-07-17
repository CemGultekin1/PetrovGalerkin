import itertools
from typing import Tuple, Union
import numpy as np
from chebyshev.core import BoundaryCondition
from chebyshev.glbsys import LocalEquationFactory, NonAllocatedGlobalSystem, SparseGlobalSystem
from chebyshev.interpolate import ErrorEstimator
from chebyshev.refinement import Refiner
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev,ChebyshevInterval
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)



def plot(gcheb:GridwiseChebyshev,filename:str,n:int = 2**8):
    x = np.linspace(0,1,n)
    y = gcheb.__call__(x)
    ytr = singfun(x)
    # logging.debug(f'y.shape = {y.shape}')
    plt.plot(x,y,'b')
    plt.plot(x,ytr,'r',alpha = 0.2,linewidth = 4)
    plt.vlines(gcheb.edges,-1.5,-.75,linestyles='solid',color = 'k',linewidth = 2)
    plt.ylim([-1,1])
    plt.savefig(filename)#)
   
    plt.close()
    
def singfun(x:NumericType):
    return np.cos((2*x)**2*np.pi)

def refinement_example():
    lof= ListOfFuns(singfun)
    flof = lof.flatten()    
    degree = 4
    gcheb = GridwiseChebyshev.from_function(flof,degree,0,1)
    
    plot(gcheb, f'before_refinements.png')
    
    refiner = Refiner(gcheb,default_refinement_scheme)
    for cynum,gcheb_ in refiner.inter_step():
        plot(gcheb_, f'after_refinements_{cynum}.png',n = 2**10)


dim = 3
def matfun(x:NumericType):
    return np.stack([np.cos(i*x*np.pi) for i in range(dim**2)],axis = 1).reshape([-1,dim,dim])
def rhsfun(x:NumericType):
    return np.stack([x**i for i in range(dim)],axis = 1).reshape([-1,dim])
def main():    
    lof = ListOfFuns(matfun,rhsfun).flatten()
    degree = 4
    gcheb = GridwiseChebyshev.from_function(lof,degree,0,1)
    refiner = Refiner(gcheb,default_refinement_scheme)
    
    bcond = BoundaryCondition(np.random.randn(dim,dim),np.random.randn(dim,dim),np.random.randn(dim,))

    leqf= LocalEquationFactory(8,bcond)
    nags = NonAllocatedGlobalSystem(dim,lof,leqf)
    # sgs = SparseGlobalSystem()
    
if __name__== '__main__':
    main()