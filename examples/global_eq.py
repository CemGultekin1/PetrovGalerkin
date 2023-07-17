import itertools
from typing import Tuple, Union
import numpy as np
from chebyshev.core import BoundaryCondition
from chebyshev.glbsys import LocalEquationFactory, NonAllocatedGlobalSystem
from chebyshev.refinement import Refiner
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev
import logging
from chebyshev.defaults import refinement_scheme
logging.basicConfig(level=logging.DEBUG)


dim = 3
def matfun(x:NumericType):
    return np.stack([np.cos(i*x*np.pi) for i in range(dim**2)],axis = 1).reshape([-1,dim,dim])
def rhsfun(x:NumericType):
    return np.stack([x**i for i in range(dim)],axis = 1).reshape([-1,dim])
def main():    
    lof = ListOfFuns(matfun,rhsfun).flatten()
    degree = 4
    gcheb = GridwiseChebyshev.from_function(lof,degree,0,1)
    refiner = Refiner(gcheb,refinement_scheme)
    
    bcond = BoundaryCondition(np.random.randn(dim,dim),np.random.randn(dim,dim),np.random.randn(dim,))

    leqf= LocalEquationFactory(8,bcond)
    nags = NonAllocatedGlobalSystem(dim,lof,leqf)
    
    

    
if __name__== '__main__':
    main()