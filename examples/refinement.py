import os
import sys
import numpy as np
from chebyshev.refinement import Refiner
from chebyshev.funs import NumericType,ListOfFuns
from chebyshev.interval import GridwiseChebyshev
import matplotlib.pyplot as plt
from chebyshev.defaults import refinement_scheme
from fldrsys.foldersys import OutputsFolders
import logging

logging.basicConfig(level=logging.DEBUG)

def singfun(x:NumericType):
    return np.cos((2*x)**4*np.pi)



def plot(gcheb:GridwiseChebyshev,foldername:str,filename:str,n:int = 2**8):
    x = np.linspace(0,1,n)
    y = gcheb.__call__(x)
    ytr = singfun(x)
    yext = 1.1
    plt.plot(x,y,'b',linewidth = 2,alpha = 0.5)
    plt.plot(x,ytr,'r',alpha = 0.2,linewidth = 4)
    height = 0.1
    plt.vlines(gcheb.edges,-yext,-yext + height,linestyles='solid',color = 'k',linewidth = 2)
    plt.vlines(gcheb.edges,-yext + height,-yext + 2*height,linestyles='solid',color = 'k',linewidth = 1)
    plt.ylim([-yext,yext])
    plt.savefig(os.path.join(foldername,filename))   
    plt.close()
    


def main():
    foldername = OutputsFolders().from_file_name(__file__).create().to_str()
    
    lof= ListOfFuns(singfun)
    flof = lof.flatten()    
    degree = 4
    gcheb = GridwiseChebyshev.from_function(flof,degree,0,1)
    
    plot(gcheb, foldername,f'refinement_{0}.png')
    
    refiner = Refiner(gcheb,refinement_scheme)
    for cynum,gcheb_ in refiner.inter_step():
        plot(gcheb_,foldername,f'refinement_{cynum}.png',n = 2**10)
    
if __name__== '__main__':
    main()