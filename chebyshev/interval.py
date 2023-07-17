from typing import List
import numpy as np
from .interpolate import coeffevl,coeffgen
from chebyshev.funs import FlatListOfFuns,NumericType

class Interval:
    def __init__(self,a:float,b:float,) -> None:
        self.interval = (a,b)
    @property
    def h(self,):
        a,b = self.interval
        return b-a
    def normalize(self,x:float):
        a,b = self.interval
        return (x - a)/(b-a)*2 -1
    def bisect(self,):
        a,b = self.interval
        m = (a+b)/2
        return Interval(a,m),Interval(m,b)
class ChebyshevCoeffs:
    def __init__(self,coeffs:np.ndarray) -> None:
        self.coeffs = coeffs
    def __call__(self,x:NumericType):
        return coeffevl(x,self.coeffs)
class ChebyshevInterval(Interval,ChebyshevCoeffs):
    def __init__(self,a:float,b:float,coeffs:np.ndarray,) -> None:
        Interval.__init__(self,a,b)
        ChebyshevCoeffs.__init__(self,coeffs)
        self.degree = coeffs.shape[-1]
        self.coeffs = coeffs.reshape([-1,self.degree])
    def to_ChebyshevCoeffs(self,):
        chebcoeff = ChebyshevCoeffs.__new__(ChebyshevCoeffs,)
        chebcoeff.coeffs = self.coeffs
        return chebcoeff
    @classmethod
    def from_function(cls,fun:FlatListOfFuns,degree:int, x0:float,x1:float,):
        coeffs = coeffgen(fun,degree,outbounds=(x0,x1))
        return ChebyshevInterval(x0,x1,coeffs,)
    def __call__(self,x:NumericType):
        xhat = self.normalize(x)
        return coeffevl(xhat,self.coeffs)    
    
    def bisect(self,fun:FlatListOfFuns):
        int0,int1 = Interval.bisect(self,)
        cint0 = ChebyshevInterval.from_function(fun,self.degree,*int0.interval)
        cint1 = ChebyshevInterval.from_function(fun,self.degree,*int1.interval)
        return cint0,cint1
        
class Grid(Interval):
    def __init__(self,x0:float,x1:float) -> None:
        super().__init__(x0,x1)
        self.edges = [x0,x1]
    def loc(self,x:NumericType):
        return np.searchsorted(self.edges,x,side = 'right') - 1
    def refine(self,i:int):
        a,b = self.edges[i],self.edges[i+1]
        m = (a+b)/2
        self.edges = self.edges[:i] + [a,m,b] + self.edges[i+2:]
        
        
class GridwiseChebyshev(Grid):
    cheblist :List[ChebyshevInterval]
    def __init__(self,fun:FlatListOfFuns,x0:float= 0,x1:float = 1) -> None:
        super().__init__(x0,x1)
        self.cheblist = []
        self.fun = fun
    @classmethod
    def from_function(cls, fun:FlatListOfFuns,degree:int ,x0:float,x1:float,):
        cint = ChebyshevInterval.from_function(fun,degree,x0,x1)
        cints = GridwiseChebyshev(fun,x0,x1)
        cints.cheblist.append(cint)
        return cints
    @property
    def hs(self,)->List[float]:
        return [cint.h for cint in self.cheblist]
    def refine(self,i:int):
        super().refine(i)
        ci = self.cheblist[i]
        ci0,ci1 = ci.bisect(self.fun)
        self.cheblist = self.cheblist[:i] +[ci0,ci1] + self.cheblist[i+1:]
    def __call__(self,x:NumericType):
        locs = self.loc(x)
        ys = []
        n = len(self.cheblist) - 1
        for i,loc in enumerate(locs):
            loc = np.minimum(loc,n)
            y = self.cheblist[loc](x[i])
            ys.append(y)
        ys = np.stack(ys,axis = 0)
        return ys
    