import itertools
from typing import Tuple, Union
import numpy as np
from chebyshev.interpolate import ErrorEstimator
from chebyshev.refinement import default_refinement_scheme,Refiner
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev,ChebyshevInterval
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

def matfun(x:NumericType):
    return np.stack([np.cos(i*x*np.pi) for i in range(9)],axis = 1).reshape([-1,3,3])
def rhsfun(x:NumericType):
    return np.stack([x**i for i in range(3)],axis = 1).reshape([-1,3])


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
    return np.cos((4*x)**2*np.pi)
def main():
    # errest = ErrorEstimator(4,8)
    # lof= ListOfFuns(singfun).flatten()    
    # chebint = ChebyshevInterval.from_function(lof,4,0,1)
    # # chebcoeffs = chebint.to_ChebyshevCoeffs()
    # logging.info(errest.evaluate(chebint.coeffs,lof,(0,1)))
    
    
    # return
    
    # lof = ListOfFuns(matfun,rhsfun)
    lof= ListOfFuns(singfun)
    flof = lof.flatten()    
    degree = 4
    gcheb = GridwiseChebyshev.from_function(flof,degree,0,1)
    
    plot(gcheb, f'before_refinements.png')
    
    refiner = Refiner(gcheb,default_refinement_scheme)
    for cynum,gcheb_ in refiner.inter_step():
        plot(gcheb_, f'after_refinements_{cynum}.png',n = 2**10)
        
    return
    
if __name__== '__main__':
    main()