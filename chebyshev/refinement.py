from typing import  Generator, List, Tuple
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
    def refinement_score(self,errs:List[float],):
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
        errs = self.test(gcheb)
        scr = self.refinement_score(errs)
        return RefinementRanking(scr)

class IntervalNumberControl(RepresentationControl):
    def __init__(self,max_num_refinements:int, max_num_intervals:int) -> None:
        self.max_num_refinements = max_num_refinements
        self.max_num_intervals =max_num_intervals
    def create_refinements(self,refinement_task:RefinementRanking,gcheb:GridwiseChebyshev,):
        n = len(gcheb.cheblist)
        if n > self.max_num_intervals:
            logging.info(f'Max number of intervals reached: {n} > {self.max_num_intervals}')
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


class RefinementScheme:
    def __init__(self,errc:ErrorControl,inc:IntervalNumberControl,grc:GridRegularityControl,max_num_cycles:int = 5) -> None:
        self.error_control = errc
        self.interval_number_control = inc
        self.grid_regularity_control = grc
        self.max_num_cycles = max_num_cycles
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
            
class Refiner:
    def __init__(self,gcheb:GridwiseChebyshev,rs:RefinementScheme) -> None:
        self.grid_cheb = gcheb
        self.refinement_scheme = rs
    def inter_step(self,):
        cy0 = 0
        for cy,rt in self.refinement_scheme.run_cycles(self.grid_cheb):
            
            if cy > cy0:
                cy0 = cy
                yield cy,self.grid_cheb
            self.run_refinement(rt)
        yield cy0,self.grid_cheb
    def run(self,):
        for cy,rt in self.refinement_scheme.run_cycles(self.grid_cheb):
            self.run_refinement(rt)
    def run_refinement(self,rt:RefinementTask):
        gcheb = rt.gridcheb
        intid = rt.current_interval_id
        gcheb.refine(intid)
        
        
default_error_control = ErrorControl(4,8,max_abs_err=1e-3)
default_interval_number_control = IntervalNumberControl(2**4,2**9)
default_grid_regularity_control = GridRegularityControl(50)
default_refinement_scheme = RefinementScheme(default_error_control,\
                default_interval_number_control,\
                default_grid_regularity_control,max_num_cycles=16)
