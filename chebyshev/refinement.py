from typing import  Any, Callable, Dict, Generator, List, Tuple
from .interval import GridwiseChebyshev,ChebyshevInterval
from .funs import FlatListOfFuns
from .interpolate import PtsWiseChebErr
import numpy as np
import logging
 
class RefinementRanking(list):
    def __init__(self,n:int) -> None:
        super().__init__()
        self.extend(list(range(n)))
        self.refinements = []
        self.offset = 0
        self.last_iter = -1
    def iterate_refs(self,):
        n = len(self)
        for _ in range(n):
            inti = self.pop(0)
            self.last_iter = inti + self.offset     
            yield self.last_iter   
        self.extend(self.refinements)
        self.refresh()
    def refresh(self,):
        self.refinements = []
        self.last_iter = -1
        self.offset = 0
    def refinement_needed(self,):
        return bool(self)
    def refinement(self,):
        self.refinements.append(self.last_iter)
        self.refinements.append(self.last_iter+1)
        self.offset = len(self.refinements)//2
class RefinmentVec:
    def __init__(self,) -> None:
        self.vec = None
        self.offset = 0
        self.last_ind = -1
    def init_vec(self,n:int):
        if self.vec is None:
            self.vec = np.ones((n,),dtype = bool)
    def assign_refinement(self,i:int,value:bool):
        self.vec[i] = value
    def kill_all_refinements(self,):
        self.vec[:] = False
    def num_refinements(self,):
        return np.sum(self.vec)
    def refined_interval(self,):        
        i = self.last_ind
        self.offset += 1
        self.vec = np.concatenate([self.vec[:i+1],\
            np.ones((1,),dtype = bool),
            self.vec[i+1:]])
    def refresh(self,):
        self.offset = 0
    def is_empty(self,):
        return not np.any(self.vec)
    def iterate_refs(self,):
        inds = np.where(self.vec,)[0]
        for i in inds:
            self.last_ind = i+ self.offset
            yield self.last_ind
        self.refresh()
        

class Control:
    tol :float = 1e-3
    def __init__(self,tol:float = -1.) -> None:
        if tol < 0:
            tol = self.__class__.tol
        self.tol = tol
    def set_error_estimator(self,)->None:...
    @property
    def name(self,):
        return self.__class__.__name__
    def __call__(self, rr: RefinementRanking,*gchebs:GridwiseChebyshev, ):...
    


class RepresentationErrorControl(Control):
    err_est : PtsWiseChebErr
    tol :float = 1e-2
    def __init__(self, tol: float = -1) -> None:
        super().__init__(tol)
    def set_error_estimator(self,err_est:PtsWiseChebErr):
        self.err_est = err_est
    def __call__(self,rr: RefinmentVec,*gchebs:GridwiseChebyshev):
        gcheb = gchebs[0]
        # rr.assign_refinement(errs >= self.tol,True)
        # np.random.seed(1)
        for i in rr.iterate_refs():
            err = self.test_interval(gcheb.cheblist[i],gcheb.fun)
            # logging.info(f'\t {i}: err = {err},')
            rr.assign_refinement(i,err > self.tol,)
    def test(self,gridwisecheb:GridwiseChebyshev)->List[float]:
        errs = []
        for chebint in gridwisecheb.cheblist:
            errs.append(self.test_interval(chebint,gridwisecheb.fun))
        return errs
    def test_interval(self,chebint:ChebyshevInterval,fun:FlatListOfFuns)->float:
        return self.err_est.evaluate(chebint.coeffs,fun,chebint.interval)

class MaxNumIntervalControl(Control):
    tol:float = 1e2
    def __call__(self,rr: RefinmentVec,*gchebs:GridwiseChebyshev,):
        gcheb = gchebs[0]
        num = gcheb.num_interval
        if num > self.tol:
            rr.kill_all_refinements()
            return
        x =  rr.num_refinements()
        if num + x < self.tol:
            return
        permitted = int(np.ceil(self.tol - x))
        hs = gcheb.hs
        hs[~ rr.vec] = 0
        hsi = np.argsort(hs)[::-1]
        permits = hsi[:permitted]
        rr.kill_all_refinements()
        rr.assign_refinement(permits,True)

class GridConditionControl(Control):
    tol :float = 50.
    def __call__(self,ref:RefinmentVec,*gchebs:GridwiseChebyshev):
        gcheb = gchebs[0]
        refmask = ref.vec          
        hs = np.array(gcheb.hs)
        hs = np.where(refmask,hs/2,hs)
        gcond = hs/np.amin(hs)
        ref.assign_refinement(gcond >= self.tol,True)
        
class CustomRefiner(RefinmentVec):
    control_cls_list : Tuple[type] = ()
    control_cls_tol :Tuple[float] = ()
    controls: Tuple[Control]
    degree_increments : Tuple[int,...] 
    def __init__(self,degree_increments :Tuple[int,...] = (2,)) -> None:
        self.degree_increments = degree_increments
        self.controls = [ccl(tol = cct) for ccl,cct in zip(self.control_cls_list,self.control_cls_tol)]
        super().__init__()
    def run_controls(self,*gchebs:GridwiseChebyshev):
        self.init_vec(gchebs[0].num_interval)
        for cntrl in self.controls:
            cntrl(self,*gchebs)
        return self.is_empty()            
    def single_refinement_decision(self,i:int,gcheb:GridwiseChebyshev,follower:bool = True):        
        if follower:
            gcheb.refine(i)
            return True
        
        deg = gcheb[i].degree            
        if  deg == self.degree_increments[-1]:                       
            if deg > self.degree_increments[0]:
                gcheb.change_degree(i,self.degree_increments[0])
            gcheb.refine(i)
            self.refined_interval()
            return True
        else:
            degi = self.degree_increments.index(deg)
            deg = self.degree_increments[degi + 1]
            gcheb.change_degree(i,deg)
            return False
    def run_refinements(self,*gchebs:GridwiseChebyshev,):        
        for i in self.iterate_refs():
            bisect_flag = self.single_refinement_decision(i,gchebs[0],follower=False)
            if not bisect_flag:
                continue
            for gcheb_ in gchebs[1:]:
                self.single_refinement_decision(i,gcheb_,follower=True)
            
        
class GridControlledRefiner(CustomRefiner):
    control_cls_list :List[type] = [GridConditionControl,MaxNumIntervalControl]
    control_cls_tol :List[float] = None
    
    def __init__(self,degree_increments : Tuple[int,...] = (2,),max_num_intrvl:float = - 1,max_grid_cond:float = 50) -> None:
        self.setup(max_num_intrvl,max_grid_cond)
        super().__init__(degree_increments)
    def setup(self,*tolerances:float):
        if self.control_cls_tol is None:
            self.control_cls_tol = list(tolerances)
            
class RepresentationRefiner(GridControlledRefiner):
    control_cls_list :List[type] = [RepresentationErrorControl] + GridControlledRefiner.control_cls_list
    def __init__(self,degree_increments:Tuple[int,...] = (2,),max_rep_err:float = - 1,max_num_intrvl:float = - 1,max_grid_cond:float = 50) -> None:
        self.setup(max_rep_err,max_num_intrvl,max_grid_cond)
        ptswisecheb = PtsWiseChebErr(degree_increments[-1])
        super().__init__(degree_increments,max_num_intrvl,max_grid_cond)
        self.controls[0].set_error_estimator(ptswisecheb)

