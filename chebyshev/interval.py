import logging
from typing import List, Tuple
import numpy as np
from .interpolate import coeffevl,coeffgen
from .funs import FlatListOfFuns,NumericType,ConcatenatedVectorSeparator,EmptySeparator,NumericFunType,ListOfFuns

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
    def __init__(self,a:float,b:float,coeffs:np.ndarray,separator:ConcatenatedVectorSeparator = EmptySeparator()) -> None:
        Interval.__init__(self,a,b)
        ChebyshevCoeffs.__init__(self,coeffs)
        self.degree = coeffs.shape[0]        
        self.coeffs = coeffs.reshape([self.degree,-1])
        self.separator = separator
    @property
    def dim(self,):
        return self.coeffs.shape[1]
    def to_ChebyshevCoeffs(self,):
        chebcoeff = ChebyshevCoeffs.__new__(ChebyshevCoeffs,)
        chebcoeff.coeffs = self.coeffs
        return chebcoeff
    @classmethod
    def from_function(cls,fun:FlatListOfFuns,degree:int, x0:float,x1:float,dim:int = -1):
        if fun is None:
            assert dim >0
            coeffs = np.zeros((degree,dim))
            return ChebyshevInterval(x0,x1,coeffs,)
        coeffs = coeffgen(fun,degree-1,outbounds=(x0,x1))
        return ChebyshevInterval(x0,x1,coeffs,separator=fun.separator)
    def left_value(self,):
        return coeffevl(-1,self.coeffs)    
    def right_value(self,):
        return coeffevl(1,self.coeffs)    
    def __call__(self,x:NumericType):
        xhat = self.normalize(x)
        return coeffevl(xhat,self.coeffs)    
    def separate_funs(self,):
        if isinstance(self.separator,EmptySeparator):            
            logging.error(f'No separator assigned!')
            raise Exception
        coeffss = self.separator(self.coeffs.reshape([self.degree,-1]),axis = 1)
        return tuple(ChebyshevInterval(*self.interval,coeffs) for coeffs in coeffss)
    def separate_dims(self,splits:Tuple[int],index:int = 0 ):
        spcoeffs = np.split(self.coeffs,splits,axis = 1)
        coeffs = spcoeffs[index]
        return self.new_by_coeff(coeffs)
    def bisect(self,fun:FlatListOfFuns):
        # logging.debug(f'Refining the interval {self.interval} into two pieces')
        int0,int1 = Interval.bisect(self,)
        cint0 = ChebyshevInterval.from_function(fun,self.degree,*int0.interval,dim = self.dim)
        cint1 = ChebyshevInterval.from_function(fun,self.degree,*int1.interval,dim = self.dim)
        return cint0,cint1
    def change_degree(self,fun:FlatListOfFuns,degree:int):
        if fun is None:
            coeffs = np.zeros((degree,self.dim))
        else:
            coeffs = coeffgen(fun,degree-1,outbounds=self.interval)
        self.coeffs = coeffs
        self.degree = degree
    def new_by_coeff(self,coeffs:np.ndarray):
        coeffs = coeffs.reshape([self.degree,-1])
        return ChebyshevInterval(*self.interval,coeffs)
        
class Grid(Interval):
    def __init__(self,x0:float,x1:float) -> None:
        super().__init__(x0,x1)
        self.edges = [x0,x1]
    def loc(self,x:NumericType):
        n = len(self.edges) - 2
        location =  np.searchsorted(self.edges,x,side = 'right') - 1
        if isinstance(x,np.ndarray):
            return np.minimum(location,n)
        elif np.isscalar(x):
            return (np.minimum(location,n),)
        else:
            logging.error(f'The input {x} is neither scalar nor np.ndarray')
    def find_closest_edges(self,x:NumericType):
        return np.array(list(map(lambda x_: np.argmin(np.abs(x_ -self.edges)),x)))
    def find_touching_intervals(self,x0:float,x1:float):
        if x1<x0:
            x0,x1 = x1,x0
        x0i,x1i = (np.argmin(np.abs(np.array(self.edges) - x)) for x in (x0,x1))
        return np.arange(x0i,x1i)
    @property
    def conds(self,):
        return np.array(self.edges)/np.amin(self.edges)
    @property
    def condition_number(self,):
        return np.amax(self.conds)
    def refine(self,i:int):
        a,b = self.edges[i],self.edges[i+1]
        m = (a+b)/2
        self.edges = self.edges[:i] + [a,m,b] + self.edges[i+2:]
    @property
    def num_interval(self,):
        return len(self.edges) -1
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
        shp = None
        for i in [0,len(edge_values)-1]:
            x = edge_values[i]
            if x.size == 0:
                edge_values[i] = edge_values[1]*0
            if shp is None:
                shp = edge_values[i].size
            else:
                assert shp == edge_values[i].size
        edge_values = [ev.flatten() for ev in edge_values]
        edge_shp = np.array([ev.size for ev in edge_values])
        assert np.all(edge_shp == edge_shp[0])
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
    def separate_dims(self,splitter:List[int],index:int = 0):
        values = np.split(self.values,splitter,axis = 2)[index]
        edgeval = EdgeValues.__new__(EdgeValues,)
        edgeval.__dict__.update(self.__dict__)
        edgeval.values = values
        return edgeval
        
class GridwiseChebyshev(Grid):
    cheblist :List[ChebyshevInterval]
    def __init__(self,fun:FlatListOfFuns,x0:float= 0,x1:float = 1) -> None:        
        super().__init__(x0,x1)
        self.cheblist = []
        self.edge_values = EdgeValues(self.cheblist,)
        self.edge_values.set_up_to_date(False)
        self.fun = fun
        self.separator = fun.separator
    def get_mean_edge_values(self,):
        if not self.edge_values.up_to_date:
            self.update_edge_values(old_head_tail=True)
        return self.edge_values.values.mean(axis = 1)
    def __iter__(self,):
        return self.cheblist.__iter__()
    @property
    def dim(self,):
        return self.cheblist[0].coeffs.shape[1]
    @classmethod
    def from_single_chebyshev(self,fun:FlatListOfFuns,chebint:ChebyshevInterval):
        gcheb = GridwiseChebyshev(fun,x0 = chebint.interval[0],x1 = chebint.interval[1])
        gcheb.cheblist.append(chebint)
        return gcheb
    def new_grided_chebyshev(self,dim:int,degree:int= -1):
        gcheb = GridwiseChebyshev.__new__(GridwiseChebyshev,)
        gcheb.__dict__.update(self.__dict__)
        chebints = []
        for cheb in self.cheblist:            
            if degree <= 0:
                cint = ChebyshevInterval(*cheb.interval,cheb.coeffs[:,:dim])
            else:
                cint = ChebyshevInterval(*cheb.interval,cheb.coeffs[:degree,:dim])
            assert cint.dim == dim
            chebints.append(cint)
        gcheb.cheblist = chebints
        gcheb.fun = None
        gcheb.edge_values = EdgeValues(chebints)
        return gcheb
        
        
            
        
    @classmethod
    def from_function(cls, fun:FlatListOfFuns,degree:int ,x0:float,x1:float,):
        cint = ChebyshevInterval.from_function(fun,degree,x0,x1)
        cints = GridwiseChebyshev(fun,x0,x1)
        cints.cheblist.append(cint)
        return cints
    @classmethod
    def from_function_and_edges(cls, fun:FlatListOfFuns,degree:int ,edges:Tuple[float]):
        cints = []
        for x0,x1 in zip(edges[:-1],edges[1:]):
            cint = ChebyshevInterval.from_function(fun,degree,x0,x1)
            cints.append(cint)
        gcheb = GridwiseChebyshev(fun,edges[0],edges[-1])
        gcheb.cheblist.extend(cints)
        gcheb.edges = list(edges)
        gcheb.update_edge_values()
        return gcheb
    
    def matching_gcheb_from_functions(self, *funs:NumericFunType,):
        fun = ListOfFuns(*funs).flatten()
        cints = []
        for degree,x0,x1 in zip(self.ps,self.edges[:-1],self.edges[1:],):
            cint = ChebyshevInterval.from_function(fun,degree,x0,x1)
            cints.append(cint)
        gcheb = GridwiseChebyshev(fun,self.edges[0],self.edges[-1])
        gcheb.cheblist.extend(cints)
        gcheb.edges = list(self.edges)
        gcheb.update_edge_values()
        return gcheb
    @property
    def hs(self,)->np.ndarray:
        return np.array([cint.h for cint in self.cheblist])
    @property
    def ps(self,)->np.ndarray:
        return np.array([cint.degree for cint in self.cheblist])
    @property
    def extended_ps(self,)->np.ndarray:
        ps = self.ps
        ps =np.insert(ps,0,1)
        ps = np.insert(ps,len(ps),1)
        return ps
    def __getitem__(self,i:int)->ChebyshevInterval:
        return self.cheblist[i]
    def refine(self,i:int):
        super().refine(i)
        ci = self.cheblist[i]
        ci0,ci1 = ci.bisect(self.fun)
        self.cheblist = self.cheblist[:i] +[ci0,ci1] + self.cheblist[i+1:]
        self.edge_values.set_up_to_date(False)
    def change_degree(self,i:int,degree:int):
        chebint = self[i]
        chebint.change_degree(self.fun,degree)
    def __call__(self,x:NumericType):
        locs = self.loc(x)
        ys = []
        # n = len(self.cheblist) - 1
        if len(self.cheblist) == 1:
            return self.cheblist[0](x)
        
        for i,loc in enumerate(locs):
            
            y = self.cheblist[loc](x[i])
            ys.append(y)
        ys = np.stack(ys,axis = 0)
        return ys
    def iterate_edge_values(self,):
        for i in range(len(self.edges)):
            lef = self.edge_values.get_interval_edge(i,left= True,)
            rig = self.edge_values.get_interval_edge(i,right = True)
            if i == 0:
                cntint = (0,)
            elif i == len(self.edges) -1:
                cntint = (i-1,)
            else:
                cntint = (i-1,i)
            yield lef,rig,cntint
    def __str__(self,):
        return f'# of intervals = {len(self.cheblist)} with (max,min) separations = {np.amax(self.hs),np.amin(self.hs)}'
    def update_edge_values(self,head_edge:np.ndarray= np.empty(0,),tail_edge:np.ndarray = np.empty(0,),old_head_tail:bool = False):
        if old_head_tail: 
            head_edge = self.edge_values.get_interval_edge(0,left = True,even_if_not_up_to_date=True)
            tail_edge = self.edge_values.get_interval_edge(-1,right = True,even_if_not_up_to_date=True)        
        self.edge_values = EdgeValues(self.cheblist,head_edge=head_edge,tail_edge=tail_edge)
        self.edge_values.set_up_to_date(True)
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
        gg.fun = None
        gg.edges = self.edges
        gg.edge_values = EdgeValues(new_cheblist,head_edge=head_edge,tail_edge=tail_edge)
        gg.edge_values.set_up_to_date(True)
        return gg
    def to_solution(self,):
        head = self.edge_values.values[0,0]
        tail = self.edge_values.values[-1,1]
        coeffs = np.concatenate([chb.coeffs.flatten() for chb in self.cheblist],)
        return np.concatenate([head,coeffs,tail])
    def adopt_solution(self,solution:np.ndarray,dim:int):
        solution = solution.reshape([-1,dim])
        head_edge = solution[0]
        tail_edge = solution[-1]
        
        solution = solution[1:-1,:]
        ps = self.ps
        per_int_solution = np.split(solution,np.cumsum(ps),axis = 0)
        new_cheblist = []        
        for coeffs,chebint in zip(per_int_solution,self.cheblist):
            new_cheblist.append(chebint.new_by_coeff(coeffs))
        self.cheblist = new_cheblist
        self.fun = None
        self.edges = self.edges
        self.edge_values = EdgeValues(new_cheblist,head_edge=head_edge,tail_edge=tail_edge)
        self.edge_values.set_up_to_date(True)
        
    def separate_dims(self,splitter:List[int],index :int = 0):
        logging.debug(f'splitter:List[int] = {splitter},index :int = {index}')
        sepchebs = []
        logging.debug(f'\t\t before: n intervals = {self.num_interval}')
        for cheb in self.cheblist:
            newcheb =cheb.separate_dims(splitter,index = index)
            sepchebs.append(newcheb)
            logging.debug(f' index = {index},\n coeffs = {newcheb.coeffs}')
        gg = GridwiseChebyshev.__new__(GridwiseChebyshev,)
        gg.cheblist = sepchebs
        gg.fun = None
        gg.edges = self.edges
        gg.edge_values = self.edge_values.separate_dims(splitter,index = index)
        logging.debug(f'\t\t after : n intervals = {gg.num_interval}')
        return gg
    @classmethod
    def create_from_local_solution(cls,chebint:ChebyshevInterval,\
                        interior_solution:np.ndarray,\
                        edge_solution:np.ndarray,dim:int):
        interior_solution = interior_solution.reshape([-1,dim])
        per_int_solution = interior_solution
        new_chebint = chebint.new_by_coeff(per_int_solution)
        new_chebint.separator = None#chebint.separator
        new_cheblist = [new_chebint]
        gg = GridwiseChebyshev.__new__(GridwiseChebyshev,)
        gg.cheblist = new_cheblist
        gg.fun = chebint
        gg.edges = list(chebint.interval)
        gg.edge_values = EdgeValues([new_chebint],head_edge=edge_solution[0],tail_edge=edge_solution[1])
        return gg
        