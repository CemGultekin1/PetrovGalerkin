import logging
import numpy as np
from typing import Any, Tuple
import numpy.polynomial.chebyshev as cheb
from .funs import NumericType,NumericFunType,FlatListOfFuns
class ShapeMemory:
    def __init__(self,fun) -> None:
        self.fun = fun
        self.shape = None
    def __call__(self, x:np.ndarray, *args: Any, **kwds: Any) -> Any:
        val =self.fun(x,*args,**kwds)        
        if self.shape is None:
            self.shape = val.shape[1:]
        val = val.reshape([len(x),-1])
        return val
class ShapedCoefficients:
    def __init__(self,fun,lefthanded:bool = True ) -> None:
        self.fun = fun
        self.lefthanded = lefthanded
    def __call__(self,fun,*args,**kwargs):
        sm = ShapeMemory(fun)
        coeffs = self.fun(sm,*args,**kwargs)
        c0 = coeffs.shape[0]
        if not self.lefthanded:
            coeffs = coeffs.T
            coeffs = coeffs.reshape([*sm.shape,c0])
        else:            
            coeffs = coeffs.reshape([c0,*sm.shape])
        return coeffs
class ShapedCoeffsEval(ShapedCoefficients):
    def __call__(self,x:NumericType,coeffs:np.ndarray)->np.ndarray:
        shape = coeffs.shape
        coeffs = coeffs.reshape([-1,sum(shape[1:])])
        evals :np.ndarray= self.fun(x,coeffs)
        if self.lefthanded:
            evals = evals.T
        if isinstance(x,(float,int)):
            evals = evals.reshape(shape[1:])
        else:
            assert isinstance(x,np.ndarray)
            evals = evals.reshape([len(x),*shape[1:]])
        return evals
class IntervalBound:
    def __init__(self,fun, inbounds:Tuple[int,int] = (-1,1),outbounds:Tuple[int,int] = (-1,1)) -> None:
        self.fun = fun
        self.inbounds = inbounds
        self.outbounds = outbounds
    def __call__(self, x:NumericType,) -> Any:
        x0,x1 = self.inbounds
        x0_,x1_ = self.outbounds
        xhat = (x - x0)/(x1- x0) * (x1_ - x0_) + x0_
        return self.fun(xhat)
class IntervalBoundDecorator:
    def __init__(self,fun) -> None:
        self.fun = fun
    def __call__(self, fun:NumericFunType,*args: Any, inbounds:Tuple[int,int] = (-1,1),outbounds:Tuple[int,int] = (-1,1),**kwargs) -> np.ndarray:
        bounded_fun = IntervalBound(fun,inbounds =inbounds, outbounds=outbounds)
        return self.fun(bounded_fun,*args,**kwargs)

class ChebyshevPoints:
    def __init__(self,degree:int):
        self.degree = degree
        self.points = np.empty(self.degree+1,dtype = float)
        self.filled = False
    def get_points(self,):
        if not self.filled:
            self.gather_points()
        return self.points
    def gather_points(self,):
        class MemCallable:
            def __init__(self,) -> None:
                self.pts = np.empty(0)
            def __call__(self,x):
                self.pts = x
                return x
        foo = MemCallable()
        _ = cheb.chebinterpolate(foo,self.degree)
        self.points = foo.pts
class ChebyshevPointsCollection:
    def __init__(self,min_degree:int,max_degree:int) -> None:
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.points_dict = {deg: ChebyshevPoints(deg) for deg in range(self.min_degree,self.max_degree+1)}
    def __getitem__(self,degree:int):
        if not (degree >= self.min_degree and  degree <= self.max_degree):
            logging.error(f'Requested degree {degree} doesn\'t fall into the interval [{self.min_degree},{self.max_degree}]!')
            raise Exception
        return self.points_dict[degree].get_points()

        
        
lefthanded = True
coeffgen = IntervalBoundDecorator(ShapedCoefficients(cheb.chebinterpolate,lefthanded=lefthanded))
coeffevl = ShapedCoeffsEval(cheb.chebval,lefthanded=lefthanded)


class PtsWiseChebErr(ChebyshevPointsCollection):
    degdif :int = 3
    def __init__(self, max_degree: int) -> None:
        super().__init__(1, max_degree + self.degdif)
    def evaluate(self,coeffs:np.ndarray, lofns:FlatListOfFuns,interval:Tuple[float,float])->float:
        deg = coeffs.shape[0] - 1
        pts_ = self[deg+self.degdif]
        a,b = interval
        pts = (pts_ + 1)/2 * (b-a) + a
        truvals =lofns(pts)
        estvals = coeffevl(pts_,coeffs,)
        # logging.debug(f'pts = {pts}')
        # logging.debug(f'truvals = {truvals}')
        # logging.debug(f'estvals = {estvals}')
        return np.amax(np.abs(truvals - estvals)/(np.abs(truvals) + np.abs(estvals) + 1e-12))*2
    
def foo(x):
    return np.stack([np.ones(x.shape),x,x**2,x**3],axis = 1).reshape([-1,2,2])

def main():
    cpt = ChebyshevPoints(10)
    cpt.gather_points()
    print(cpt.points)
if __name__ == '__main__':
    main()