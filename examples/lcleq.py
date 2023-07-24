import os
import numpy as np
from solver.core import BoundaryCondition
from solver.glbsys import DenseLocalSystem
from solver.eqgen import EquationFactory
from solver.lclsolve import LocalSysAllocator
from solver.linsolve import GlobalSystemSolver,LocalSystemSolver
from chebyshev import ListOfFuns,GridwiseChebyshev,NumericType
import logging
import matplotlib.pyplot as plt
from examples.nprint import MatPrinter
from examples.plotter import GridwiseChebyshevPlotter, MultiDimGridwiseChebyshevPlotter
from fldrsys.folders import OutputsFolders
from solver.lcl2err import ResidualFunction
logging.basicConfig(level=logging.INFO)


dim = 1
def matfun(x:NumericType):
    return np.stack([np.cos((i*x/2)**2*np.pi) for i in range(dim**2)],axis = 1).reshape([-1,dim,dim])*0- 1
def rhsfun(x:NumericType):
    return np.stack([x**(i+1) for i in range(dim)],axis = 1).reshape([-1,dim])
def main():    
    lof = ListOfFuns(matfun,rhsfun).flatten()
    degree = 4
    max_degree = degree
    gcheb = GridwiseChebyshev.from_function(lof,degree,0,1)
    np.random.seed(0)
    bcond = BoundaryCondition(np.eye(dim),np.random.randn(dim,dim)*0,np.random.randn(dim,)*0 + 1)

    leqf= EquationFactory(dim,max_degree,bcond)
    lsa = LocalSysAllocator(dim,leqf)

    lclcheb = gcheb.cheblist[0]
    blocks,rhs = lsa.get_single_interval_blocks(lclcheb)
    sgs = DenseLocalSystem(blocks,rhs)
    printer = MatPrinter(width=4,decimals=3)
    matstr = printer.to_str(sgs.mat)
    logging.info(f'\n\n mat = \n {matstr}')
    lss = LocalSystemSolver(sgs)
    lss.solve()
    
    logging.info(f'\n\n lss.solution =\n\n{printer.to_str(lss.solution)}')
    
    degree = lclcheb.degree
    solution = GridwiseChebyshev.create_from_local_solution(lclcheb,lss.interior_solution,lss.edge_solution,dim**2*degree)
    
    foldername = OutputsFolders().from_file_name(__file__).create().to_str()    
    
    gcp = MultiDimGridwiseChebyshevPlotter()
    for i,(fig,axs) in enumerate(gcp.draw(solution)):  
        axs.set_title(i)      
        filename = f'solution_{i}.png'
        plt.savefig(os.path.join(foldername,filename))
        plt.close()
    
    # return
    rf = ResidualFunction(dim,lclcheb,lss)

    xs= np.linspace(1e-3,1-1e-3,100)
    ys = rf(xs)
    print(ys.shape)
    print(xs.shape)
    ys = ys.reshape(len(xs),-1)
    for i in range(ys.shape[1]):
        plt.plot(xs,ys[:,i])
        folder = OutputsFolders().from_file_name(__file__).create().to_str()
        plt.savefig(os.path.join(folder,f'res-{i}.png'))
        plt.close()
    return
    
    
    gss = GlobalSystemSolver(sgs)
    gss.solve()

    solution = gss.get_wrapped_solution(gcheb)
    logging.debug(f'solution.ps = {solution.ps}, solution.hs = {solution.hs}')
    
    
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