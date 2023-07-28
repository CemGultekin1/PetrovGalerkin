from typing import List
from solver.design import LossFunction
import numpy as np
from hybrid.eqtns import Fingerprints,HybridStateSolution
class CramerRaoBound(LossFunction):
    def __init__(self,time_edges:np.ndarray,variables_inds:List[int]):
        self.time_edges = time_edges
        self.variables_inds = variables_inds
    def fischer_mat(self,fingerprints:Fingerprints):
        fischer = fingerprints.values.T@fingerprints.values + 1e-2*np.eye(fingerprints.values.shape[1])
        return fischer
    def crb_mat(self,fingerprints:Fingerprints):
        return np.linalg.inv(self.fischer_mat(fingerprints))
    def __call__(self,fingerprints:Fingerprints):        
        return np.trace(self.crb_mat(fingerprints))
    def gradient(self,u:HybridStateSolution):
        fingerprints = Fingerprints(u)
        
        
        crbmat = self.crb_mat(fingerprints)
        gf = crbmat @ fingerprints.values.T
        grad = np.zeros(fingerprints.values.shape)
        for vind in self.variables_inds:
            grad += -np.outer(gf[vind,:],crbmat[vind,:])   
            
             
        g =  fingerprints.state_avg_edges_derivative_inner_product(grad)
        dgdx = fingerprints.design_derivative_inner_product(grad)
        return g,dgdx

        
class BasicLoss(LossFunction):
    def __init__(self,time_edges:np.ndarray,variables_inds:List[int]):
        self.time_edges = time_edges
        self.variables_inds = variables_inds
    def fischer_mat(self,fingerprints:Fingerprints):
        fischer = np.sum(fingerprints.values)
        return fischer
    def __call__(self,u:HybridStateSolution):        
        return self.fischer_mat(Fingerprints(u))
    def gradient(self,u:HybridStateSolution):
        fingerprints = Fingerprints(u)
        
        grad = fingerprints.values             
        g =  fingerprints.state_avg_edges_derivative_inner_product(grad)
        dgdx = fingerprints.design_derivative_inner_product(grad)
        return g,dgdx
