import os
import numpy as np
from chebyshev.core import BoundaryCondition
from chebyshev.glbsys import SparseGlobalSystem,GlobalSysAllocator
from chebyshev.eqgen import LocalEquationFactory
from chebyshev.linsolve import GlobalSystemSolver
from chebyshev.refinement import Refiner
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev
import logging
from chebyshev.defaults import refinement_scheme
import matplotlib.pyplot as plt
from examples.nprint import MatPrinter
from examples.plotter import GridwiseChebyshevPlotter
from fldrsys.foldersys import OutputsFolders
logging.basicConfig(level=logging.INFO)


dim = 1
def matfun(x:NumericType):
    return np.stack([np.cos((i*x/2)**2*np.pi) for i in range(dim**2)],axis = 1).reshape([-1,dim,dim])*0 - 10
def rhsfun(x:NumericType):
    return np.stack([x**(i+1) for i in range(dim)],axis = 1).reshape([-1,dim])
def main():    
    lof = ListOfFuns(matfun,rhsfun).flatten()
    degree = 5
    max_degree = degree
    gcheb = GridwiseChebyshev.from_function(lof,degree,0,1)
    np.random.seed(0)
    bcond = BoundaryCondition(np.eye(dim),np.random.randn(dim,dim)*0,np.random.randn(dim,)*0 + 1)

    leqf= LocalEquationFactory(dim,max_degree,bcond)
    nags = GlobalSysAllocator(dim,lof,leqf)

    blocks = nags.create_blocks(gcheb)
    sgs = SparseGlobalSystem(blocks)
    
    
    
    gss = GlobalSystemSolver(sgs)
    gss.solve()

    solution = gss.get_wrapped_solution(gcheb)
    logging.debug(f'solution.ps = {solution.ps}, solution.hs = {solution.hs}')
    
    printer = MatPrinter(width=4,decimals=3)
    matstr = printer.to_str(gss.mat.toarray())
    rhsstr = printer.to_str(gss.rhs)
    solstr = printer.to_str(gss.solution)
    logging.debug(f'\n\nmat = \n{matstr}\n\n rhs = \n{rhsstr},\n\n sol = \n{solstr}')
   

    plt.spy(sgs.mat,markersize=1)
    folder = OutputsFolders().from_file_name(__file__).create().to_str()
    plt.savefig(os.path.join(folder,'spy.png'))
    plt.close()
    
    
    foldername = OutputsFolders().from_file_name(__file__).create().to_str()
    filename = 'solution.png'
    
    gcp = GridwiseChebyshevPlotter()
    fig = gcp.draw(solution)
    fig.savefig(os.path.join(foldername,filename))
    
if __name__== '__main__':
    main()