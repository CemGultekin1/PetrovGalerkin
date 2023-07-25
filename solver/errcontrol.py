from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Union
from chebyshev import GridwiseChebyshev,Control,GridControlledRefiner
import numpy as np
from .lclerr import LocalErrorEstimate
    

            
class LclErrControl(Control):
    default_tol : float = 1e-2
    def __init__(self, tol: float = -1) -> None:
        super().__init__(tol)
        self.err_est = None
    def set_error_estimator(self,err_est:LocalErrorEstimate):
        self.err_est = err_est
    def __call__(self,ref:GridControlledRefiner, *gchebs:GridwiseChebyshev,):
        gcheb,gcheb1 = gchebs[0:2]
        for i in ref.iterate_refs():
            cheb = gcheb[i]
            cheb1 = gcheb1[i]
            err = self.err_est.interval_error(cheb,cheb1)
            logging.info(f'interval({i}) = {err},{self.tol}')
            ref.assign_refinement(i,err > self.tol)
                
        
class SolutionRefiner(GridControlledRefiner):
    control_cls_list :List[type] = [LclErrControl] + GridControlledRefiner.control_cls_list
    def __init__(self,lclerrest:LocalErrorEstimate,degree_increments:Tuple[int,...] = (2,),max_lcl_err:float = - 1,max_num_intrvl:float = - 1,max_grid_cond:float = 50) -> None:
        self.setup(max_lcl_err,max_num_intrvl,max_grid_cond)
        super().__init__(degree_increments,max_num_intrvl,max_grid_cond)
        self.controls[0].set_error_estimator(lclerrest)
