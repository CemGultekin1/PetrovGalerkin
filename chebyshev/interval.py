import logging
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
        self.degree = coeffs.shape[0]        
        self.coeffs = coeffs.reshape([self.degree,-1])
    def to_ChebyshevCoeffs(self,):
        chebcoeff = ChebyshevCoeffs.__new__(ChebyshevCoeffs,)
        chebcoeff.coeffs = self.coeffs
        return chebcoeff
    @classmethod
    def from_function(cls,fun:FlatListOfFuns,degree:int, x0:float,x1:float,):
        coeffs = coeffgen(fun,degree-1,outbounds=(x0,x1))
        return ChebyshevInterval(x0,x1,coeffs,)
    def left_value(self,):
        return coeffevl(-1,self.coeffs)    
    def right_value(self,):
        return coeffevl(1,self.coeffs)    
    def __call__(self,x:NumericType):
        xhat = self.normalize(x)
        return coeffevl(xhat,self.coeffs)    
    def separate_funs(self,fun:FlatListOfFuns):
        coeffss = fun.separate_vector(self.coeffs.reshape([self.degree,-1]),axis = 1)
        return tuple(ChebyshevInterval(*self.interval,coeffs) for coeffs in coeffss)
        
    def bisect(self,fun:FlatListOfFuns):
        # logging.debug(f'Refining the interval {self.interval} into two pieces')
        int0,int1 = Interval.bisect(self,)
        cint0 = ChebyshevInterval.from_function(fun,self.degree,*int0.interval)
        cint1 = ChebyshevInterval.from_function(fun,self.degree,*int1.interval)
        return cint0,cint1
    def new_by_coeff(self,coeffs:np.ndarray):
        coeffs = coeffs.reshape([self.degree,-1])
        return ChebyshevInterval(*self.interval,coeffs)
        
class Grid(Interval):
    def __init__(self,x0:float,x1:float) -> None:
        super().__init__(x0,x1)
        self.edges = [x0,x1]
    def loc(self,x:NumericType):
        location =  np.searchsorted(self.edges,x,side = 'right') - 1
        if isinstance(x,np.ndarray):
            return location
        elif np.isscalar(x):
            return (location,)
        else:
            logging.error(f'The input {x} is neither scalar nor np.ndarray')
    def refine(self,i:int):
        a,b = self.edges[i],self.edges[i+1]
        m = (a+b)/2
        self.edges = self.edges[:i] + [a,m,b] + self.edges[i+2:]

class EdgeValues:
    def __init__(self,cheblist:List[ChebyshevInterval],head_edge:np.ndarray = np.empty(0),tail_edge:np.ndarray = np.empty(0)):
        self.values = np.empty((0,2,0))
        self.up_to_date = True
        if not bool(cheblist):
            return
        edge_values = [head_edge,]
        for chn in cheblist:
            edge_values.append(chn.left_value())
            edge_values.append(chn.right_value())
        edge_values.append(tail_edge)
        for i in [0,len(edge_values)-1]:
            x = edge_values[i]
            if x.size == 0:
                edge_values[i] = edge_values[1]*0

        edge_values = [ev.reshape([1,-1]) for ev in edge_values]
        edge_values = np.stack(edge_values,axis = 0)
        edge_values = edge_values.reshape(edge_values.shape[0]//2,2,edge_values.shape[1]) # time x l/r x dim
        self.values = edge_values  
        self.up_to_date = True
    def set_up_to_date(self,flag:bool):
        self.up_to_date = flag
    def get_interval_edge(self,edgenum:int,left:bool = False,right:bool = False,even_if_not_up_to_date:bool = False):
        if not even_if_not_up_to_date:
            if not self.up_to_date:
                raise Exception
        index = 0 if left else 1
        return self.values[edgenum,index]
        
class GridwiseChebyshev(Grid):
    cheblist :List[ChebyshevInterval]
    def __init__(self,fun:FlatListOfFuns,x0:float= 0,x1:float = 1) -> None:
        super().__init__(x0,x1)
        self.cheblist = []
        self.edge_values = EdgeValues(self.cheblist,)
        self.edge_values.set_up_to_date(False)
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
    @property
    def ps(self,)->List[int]:
        return [cint.degree for cint in self.cheblist]
    def refine(self,i:int):
        super().refine(i)
        ci = self.cheblist[i]
        ci0,ci1 = ci.bisect(self.fun)
        self.cheblist = self.cheblist[:i] +[ci0,ci1] + self.cheblist[i+1:]
        self.edge_values.set_up_to_date(False)
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
    def __str__(self,):
        return f'# of intervals = {len(self.cheblist)} with (max,min) separations = {np.amax(self.hs),np.amin(self.hs)}'
    def update_edge_values(self,head_edge:np.ndarray= np.empty(0,),tail_edge:np.ndarray = np.empty(0,)):
        if head_edge.size == 0 or tail_edge.size == 0: 
            head_edge = self.edge_values.get_interval_edge(0,left = True,even_if_not_up_to_date=True)
            tail_edge = self.edge_values.get_interval_edge(-1,right = True,even_if_not_up_to_date=True)        
        self.edge_values = EdgeValues(self.cheblist,head_edge=head_edge,tail_edge=tail_edge)
    def create_from_solution(self,solution:np.ndarray,dim:int):
        solution = solution.reshape([-1,dim])
        head_edge = solution[0]
        tail_edge = solution[-1]
        
        solution = solution[1:-1,:]
        ps = self.ps
        per_int_solution = np.split(solution,np.cumsum(ps),axis = 0)
        new_cheblist = []
        
        for coeffs,chebint in zip(per_int_solution,self.cheblist):
            new_cheblist.append(chebint.new_by_coeff(coeffs))
        gg = GridwiseChebyshev.__new__(GridwiseChebyshev,)
        gg.cheblist = new_cheblist
        gg.fun = self
        gg.edges = self.edges
        gg.edge_values = EdgeValues(new_cheblist,head_edge=head_edge,tail_edge=tail_edge)
        return gg
        