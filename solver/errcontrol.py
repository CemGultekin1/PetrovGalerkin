from collections import defaultdict
import logging
from typing import Dict, Union
from chebyshev import GridwiseChebyshev ,CustomRefiner,NumericType
import numpy as np

class Tolerance:
    def __init__(self,value:float) -> None:
        self.value = value
    def satisfy(self,x:NumericType):
        return x < self.value

class Control: 
    default_tol:float = 0
    def __init__(self,tol:Union[float,Tolerance]  = np.nan) -> None:
        if isinstance(tol,(float,int)):
            if np.isnan(tol):
                tol = self.default_tol
            tol = Tolerance(tol)
        self.tol = tol
    @property
    def name(self,):
        return self.__class__.__name__
    
class GridConditionControl(Control):
    default_tol :float = 50.
    @staticmethod
    def get_last_score(**kwargs)->np.ndarray:
        lastkey = list(kwargs.keys())[-1]
        return kwargs[lastkey]
    def __call__(self,gcheb:GridwiseChebyshev,**kwargs):
        scrs = self.get_last_score(**kwargs)
        hs =np.array(gcheb.hs)
        newhs = np.where(scrs > 0,hs/2,hs)
        condnum = newhs/np.amin(newhs)
        sts = self.tol.satisfy(condnum)
        if np.all(sts):
            return scrs
        maxscrs = np.amax(scrs)
        if maxscrs == - np.inf:
            maxscrs = 1
            
        return np.where(~sts,maxscrs,scrs)
        
class EdgeErrorControl(Control):
    default_tol :float = 1e-1
    def __call__(self,gcheb:GridwiseChebyshev):
        error_collection = defaultdict(lambda : 0)
        for l,r,cints in gcheb.iterate_edge_values():
            err = np.linalg.norm(l - r)
            for cint in cints: 
                error_collection[cint] = max(err,error_collection[cint])
        scrs = np.array(list(error_collection.values()))
        maxscr = np.amax(scrs)
        logging.info(f'\t\t Max edge-err = {maxscr},\t satisfies tolerance({self.tol.value}) = {self.tol.satisfy(maxscr)}')
        scrs = np.where(self.tol.satisfy(scrs),-np.inf,scrs)
        return scrs

class ControlRefinement(CustomRefiner):
    def __init__(self) -> None:
        super().__init__()
        self.add_control(EdgeErrorControl)
        self.add_merger(GridConditionControl)
    def add_control(self, controlclass:type,):
        control :Control= controlclass()        
        return super().add_control(control, control.name)
    def add_merger(self, controlclass:type,):
        control :Control= controlclass()        
        return super().add_merger(control, control.name)
        
            
            