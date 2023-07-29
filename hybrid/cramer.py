from typing import List
from solver.design import LossFunction
import numpy as np
from hybrid.fingerprints import Fingerprints
class CramerRaoBound(LossFunction):
    def __init__(self,time_edges:np.ndarray,variables_inds:List[int]):
        self.time_edges = time_edges
        self.variables_inds = variables_inds
    def fischer_mat(self,fingerprints:Fingerprints):
        fischer = fingerprints.values.T@fingerprints.values # + np.eye(fingerprints.values.shape[1])
        return fischer
    def crb_mat(self,fingerprints:Fingerprints):
        fmat = self.fischer_mat(fingerprints)
        return np.linalg.inv(fmat)
    # def __call__(self,u:HybridStateSolution):        
    #     fingerprints = Fingerprints(u)
    def __call__(self,fingerprints:Fingerprints):  
        mat = self.crb_mat(fingerprints)
        dmat = np.diag(mat)
        dmat = dmat[np.array(self.variables_inds)]
        return np.sum(dmat)
    # def gradient(self,u:HybridStateSolution):
    #     fingerprints = Fingerprints(u)
    def gradient(self,fingerprints:Fingerprints):
        
        
        crbmat = self.crb_mat(fingerprints)
        gf = crbmat @ fingerprints.values.T
        grad = np.zeros(fingerprints.values.shape)
        for vind in self.variables_inds:
            grad += -2*np.outer(gf[vind,:],crbmat[vind,:])   
        return grad

        
