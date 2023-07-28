from typing import Tuple
from solver.design import DesignSupport
from hybrid.eqtns import Fingerprints,HybridStateSolution
import numpy as np
class ThetaDesign(DesignSupport,HybridStateSolution):
    '''
    we treat it as though there 2 types of theta
    theta in free precession
    theta during rf-pulse
    '''
    def __init__(self,hss:HybridStateSolution) -> None:
        self.__dict__.update(hss.__dict__)
        fng = Fingerprints(hss)
        self.edges = fng.edges        
        
        edges = self.tr*np.arange(self.num_fp+1)
        edges0= edges[1:-1] - self.trf_seq/2
        edges1 = edges[1:-1] + self.trf_seq/2
        edges = np.concatenate([edges0,edges1,edges])
        edges.sort()
        self.design_division_edges = edges
    def __len__(self,):
        return len(self.design_division_edges) - 1
    def get_design_span(self,design_ind:int,)->Tuple[float,float]:
        return self.design_division_edges[design_ind], self.design_division_edges[design_ind + 1]
    def grad_unite(self,theta1_grad:np.ndarray,theta2_grad:np.ndarray,dldesign:np.ndarray):
        '''
        
         0 0 0 | 1 1 | 1 1 | 0 0 0
        '''
        n = len(theta1_grad)
        fptheta = theta1_grad[::3]
        left_rf = theta1_grad[1:3:n] + theta1_grad[2:3:n]
        right_rf = theta2_grad[1:3:n] + theta2_grad[2:3:n]        
        grad = np.zeros(self.theta_seq.shape)
        if not len(grad) == len(fptheta):
            print(f'len(self.theta_seq),len(theta1_grad) = {len(self.theta_seq),len(fptheta)}')
            raise Exception
        grad += fptheta
        grad[1:] += right_rf
        grad[:-1] += left_rf
        return grad + dldesign
        
        
        