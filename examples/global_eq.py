import os
import numpy as np
from chebyshev.core import BoundaryCondition
from chebyshev.glbsys import LocalEquationFactory, NonAllocatedGlobalSystem, SparseGlobalSystem
from chebyshev.refinement import Refiner
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev
import logging
from chebyshev.defaults import refinement_scheme
import matplotlib.pyplot as plt
from fldrsys.foldersys import OutputsFolders
logging.basicConfig(level=logging.INFO)


dim = 2
def matfun(x:NumericType):
    return np.stack([np.cos((i*x/2)**2*np.pi) for i in range(dim**2)],axis = 1).reshape([-1,dim,dim])
def rhsfun(x:NumericType):
    return np.stack([x**i for i in range(dim)],axis = 1).reshape([-1,dim])
def main():    
    lof = ListOfFuns(matfun,rhsfun).flatten()
    degree = 5
    gcheb = GridwiseChebyshev.from_function(lof,degree,0,1)
    refiner = Refiner(gcheb,refinement_scheme)
    logging.info(str(gcheb))
    refiner.run()    
    logging.info(str(gcheb))
    bcond = BoundaryCondition(np.random.randn(dim,dim),np.random.randn(dim,dim),np.random.randn(dim,))

    leqf= LocalEquationFactory(dim,8,bcond)
    nags = NonAllocatedGlobalSystem(dim,lof,leqf)

    blocks = nags.create_blocks(gcheb)
    sgs = SparseGlobalSystem(blocks)
    plt.spy(sgs.mat,markersize=1)
    folder = OutputsFolders().from_file_name(__file__).create().to_str()
    plt.savefig(os.path.join(folder,'spy.png'))
    plt.close()

    
if __name__== '__main__':
    main()