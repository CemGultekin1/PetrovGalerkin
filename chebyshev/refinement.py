from typing import  Callable, Dict, Generator, List
from .interval import GridwiseChebyshev,ChebyshevInterval
from .funs import FlatListOfFuns
from .interpolate import ErrorEstimator
import numpy as np
import logging


class RefinementRanking:
    def __init__(self,interval_scores:np.ndarray) -> None:
        if isinstance(interval_scores,(list,tuple)):
            interval_scores = np.array(interval_scores)
        interval_ids = np.arange(len(interval_scores))
        interval_ids = interval_ids[interval_scores>0]
        interval_scores = interval_scores[interval_scores>0]
        argi = np.argsort(interval_scores)[::-1]
        self.interval_ids = interval_ids[argi]
        self.interval_scores = interval_scores[argi]
    def is_empty(self,):
        return len(self.interval_ids)== 0
    def refined_interval_index(self,i:int):
        self.interval_ids+=np.where(self.interval_ids>i,1,0)        
    def clean_all(self,):
        self.interval_scores = np.empty(0,dtype = float)     
        self.interval_ids = np.empty(0,dtype = int)
    def iterate_refinement_needing(self,)->Generator[int,None,None]:
        for i in range(len(self.interval_ids)):
            iid,isc = self.interval_ids[i],self.interval_scores[i]
            if isc==0:
                continue
            yield iid
            self.refined_interval_index(iid)
    
    def prune_by(self,n:int):
        if n == 0:
            self.clean_all()
            return
        if n > len(self.interval_ids):
            return 
        self.interval_scores = self.interval_scores[:n]
        self.interval_ids = self.interval_ids[:n]
    def __str__(self,):
        return f'Number of refinement needed intervals = {len(self.interval_ids)}'
class RepresentationControl:
    def create_refinements(self,gcheb:GridwiseChebyshev)->RefinementRanking:...
    
    
class ErrorControl(RepresentationControl):
    def __init__(self, min_degree: int, max_degree: int,max_abs_err:float = 1e-3) -> None:
        self.errest = ErrorEstimator(min_degree,max_degree)
        self.max_abs_err = max_abs_err
    def refinement_score(self,gcheb:GridwiseChebyshev,):
        errs = self.test(gcheb)
        errs = np.array(errs)
        errs = np.where(errs < self.max_abs_err,0,errs)
        return errs
    def test(self,gridwisecheb:GridwiseChebyshev)->List[float]:
        errs = []
        for chebint in gridwisecheb.cheblist:
            errs.append(self.test_interval(chebint,gridwisecheb.fun))
        return errs
    def test_interval(self,chebint:ChebyshevInterval,fun:FlatListOfFuns)->float:
        return self.errest.evaluate(chebint.coeffs,fun,chebint.interval)
    def create_refinements(self,gcheb:GridwiseChebyshev,):        
        scr = self.refinement_score(gcheb)
        return RefinementRanking(scr)

class IntervalNumberControl(RepresentationControl):
    def __init__(self,max_num_refinements:int, max_num_intervals:int) -> None:
        self.max_num_refinements = max_num_refinements
        self.max_num_intervals =max_num_intervals
    def create_refinements(self,refinement_task:RefinementRanking,gcheb:GridwiseChebyshev,):
        n = len(gcheb.cheblist)
        if n > self.max_num_intervals:
            refinement_task.clean_all()
            return refinement_task
        diff = self.max_num_intervals - n
        refinement_task.prune_by(diff)
        return refinement_task
    
class GridRegularityControl(RepresentationControl):
    def __init__(self,condition_bound:float) -> None:
        self.condition_bound = condition_bound
    def create_refinements(self,gcheb:GridwiseChebyshev):
        hs = np.array(gcheb.hs)
        reg = hs/np.amin(hs)
        conditioning_score = np.where(reg > self.condition_bound,reg/self.condition_bound,0)
        return RefinementRanking(conditioning_score)
class RefinementTask:
    def __init__(self,gcheb:GridwiseChebyshev,interval_id:int) -> None:
        self.gridcheb = gcheb
        self.current_interval_id = interval_id
        
    
class RepresetationRefinementScheme:
    def __init__(self,errc:ErrorControl,inc:IntervalNumberControl,grc:GridRegularityControl,max_num_cycles:int = 5) -> None:
        self.error_control = errc
        self.interval_number_control = inc
        self.grid_regularity_control = grc
        self.max_num_cycles = max_num_cycles
    # def max_min_degrees(self,):
    def run_cycles(self,gcheb:GridwiseChebyshev):
        no_more_refinement = False
        num_cycle = 0
        while not no_more_refinement and num_cycle < self.max_num_cycles:
            no_more_refinement = True
            logging.debug(f'Running cycle #{num_cycle}')
            for rt in self.cycle(gcheb):
                
                no_more_refinement = False
                yield num_cycle,rt
            num_cycle+=1
    def cycle(self,gcheb:GridwiseChebyshev):
        rt = self.error_control.create_refinements(gcheb)
        logging.debug(f'Error control: \t\t\t{str(rt)}')
        rt = self.interval_number_control.create_refinements(rt,gcheb)
        logging.debug(f'Interval number control: \t\t{str(rt)}')
        for ii in rt.iterate_refinement_needing():
            yield RefinementTask(gcheb,ii)
        rt = self.grid_regularity_control.create_refinements(gcheb)
        logging.debug(f'Grid regularity control: \t\t{str(rt)}')
        for ii in rt.iterate_refinement_needing():
            yield RefinementTask(gcheb,ii)
    def inter_step(self,gcheb:GridwiseChebyshev):
        cy0 = 0
        for cy,rt in self.run_cycles(gcheb):
            
            if cy > cy0:
                cy0 = cy
                yield cy,gcheb
            self.run_refinement(rt)
        yield cy0,gcheb
    def run(self,gcheb:GridwiseChebyshev):
        for _,rt in self.run_cycles(gcheb):
            self.run_refinement(rt)
    def run_refinement(self,rt:RefinementTask):
        gcheb = rt.gridcheb
        intid = rt.current_interval_id
        gcheb.refine(intid)
        
        
control = 'control'
merger = 'merger'
class CustomRefiner:
    def __init__(self,) -> None:
        self.counter = 0
        self.controls  = {}
        self.scrs = {}
    def add_control(self,cntrlfun:Callable[[GridwiseChebyshev],np.ndarray],name:str):
        self.controls[self.counter] = (cntrlfun,name,control)
        self.counter+=1
    def add_merger(self,mergerfun:Callable[[GridwiseChebyshev,Dict[str,np.ndarray]],np.ndarray],name:str):
        self.controls[self.counter] = (mergerfun,name,merger)
        self.counter+=1
    def run_controls(self,gcheb:GridwiseChebyshev):
        scrs = {}
        for fun,name,tag in self.controls.values():
            if tag == control:
                scrs[name] = fun(gcheb)
            elif tag == merger:
                scrs[name] = fun(gcheb,**scrs)
            else:
                raise Exception
        self.scrs = scrs
        return not self.is_refinement_needed()
    # def refine(self,gcheb:GridwiseChebyshev,):
    #     self.run_controls(gcheb)
    #     self.run_refinements(gcheb,)
    #     return gcheb
    @property
    def final_score(self,):
        if not bool(self.scrs):
            return -np.ones(1)*np.inf
        last_key = list(self.scrs.keys())[-1]
        return self.scrs[last_key]
    def is_refinement_needed(self,):
        fscr = self.final_score
        if np.all(fscr <= 0):
            return False
        return True
            
    def run_refinements(self,gcheb:GridwiseChebyshev,):
        if not self.is_refinement_needed():
            return gcheb
        scrs = self.final_score
        scris = np.argsort(scrs)[::-1]
        intervals =scris.copy()
        scrs = scrs[scris]
        for i,scr in enumerate(scrs):
            if scr <= 0:
                continue
            gcheb.refine(intervals[i])
            intervals+=np.where(intervals>intervals[i],1,0)
        self.scrs = {}
        return gcheb
            
