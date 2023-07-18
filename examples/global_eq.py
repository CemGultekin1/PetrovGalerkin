import os
import numpy as np
from chebyshev.core import BoundaryCondition
from chebyshev.glbsys import LocalEquationFactory, GlobalSysAllocator, SparseGlobalSystem
from chebyshev.linsolve import GlobalSystemSolver
from chebyshev.refinement import Refiner
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev
import logging
from chebyshev.defaults import refinement_scheme
import matplotlib.pyplot as plt
from examples.nprint import MatPrinter
from fldrsys.foldersys import OutputsFolders
logging.basicConfig(level=logging.INFO)


dim = 1
def matfun(x:NumericType):
    return np.stack([np.cos((i*x/2)**2*np.pi) for i in range(dim**2)],axis = 1).reshape([-1,dim,dim])*0
def rhsfun(x:NumericType):
    return np.stack([x**i for i in range(dim)],axis = 1).reshape([-1,dim])*0
def main():    
    lof = ListOfFuns(matfun,rhsfun).flatten()
    degree = 2
    max_degree = degree
    gcheb = GridwiseChebyshev.from_function(lof,degree,0,1)
    # refiner = Refiner(gcheb,refinement_scheme)
    logging.info(str(gcheb))
    # refiner.run()    
    gcheb.refine(0)
    logging.info(str(gcheb))
    np.random.seed(0)
    bcond = BoundaryCondition(np.eye(dim),np.random.randn(dim,dim)*0,np.random.randn(dim,)*0)

    leqf= LocalEquationFactory(dim,max_degree,bcond)
    nags = GlobalSysAllocator(dim,lof,leqf)

    blocks = nags.create_blocks(gcheb)
    sgs = SparseGlobalSystem(blocks)
    
    
    
    gss = GlobalSystemSolver(sgs)
    gss.solve()
    solution = gss.get_wrapped_solution(gcheb)
    logging.info(f'solution.ps = {solution.ps}, solution.hs = {solution.hs}')
    
    printer = MatPrinter(width=4,decimals=1)
    matstr = printer.to_str(gss.mat.toarray())
    rhsstr = printer.to_str(gss.rhs)
    solstr = printer.to_str(gss.solution)
    logging.info(f'\n\nmat = \n{matstr},\n\n rhs = \n{rhsstr},\n\n sol = \n{solstr}')
    return

    plt.spy(sgs.mat,markersize=1)
    folder = OutputsFolders().from_file_name(__file__).create().to_str()
    plt.savefig(os.path.join(folder,'spy.png'))
    plt.close()
    
    n = 2**8
    x = np.linspace(0,1,n)
    y = solution(x)
    # yext = 1.1
    plt.plot(x,y,)#'b',linewidth = 2,alpha = 0.5)

    foldername = OutputsFolders().from_file_name(__file__).create().to_str()
    filename = 'solution.png'
    
    # height = 0.1
    # plt.vlines(gcheb.edges,-yext,-yext + height,linestyles='solid',color = 'k',linewidth = 2)
    # plt.vlines(gcheb.edges,-yext + height,-yext + 2*height,linestyles='solid',color = 'k',linewidth = 1)
    # plt.ylim([-yext,yext])
    plt.savefig(os.path.join(foldername,filename))   
    plt.close()

    
if __name__== '__main__':
    main()