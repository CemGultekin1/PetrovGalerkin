from .quadrature import InnerProducts
import numpy as np

class ResidualNorm(InnerProducts):
    def __init__(self, degree: int) -> None:
        super().__init__(2*degree)
        self.orthogonals = {deg:OrthogonalProjection(deg) for deg in range(degree,2*degree)}
    def fillup(self):
        super().fillup()
        self.orthogonals = {deg:op.compute() for deg,op in self.orthogonals.items()}    
    def residual_norm(self,coeffs:np.ndarray,)->float:
        deg = coeffs.shape[0]
        coeffs = coeffs.reshape([deg,-1])
        hnorm = self.orthogonals[deg]@coeffs
        return np.linalg.norm(hnorm,'fro')
        
        
        
        
class OrthogonalProjection:
    def __init__(self,degree:int,innprod:InnerProducts) -> None:
        self.degree = degree
        self.innprod = innprod
    def compute(self,):
        p = self.degree//2
        q = self.degree
        Q = self.innprod.dub_quads[:q,:q]
        EQE = Q[:p,:p]
        EQ = Q[:p,:]
        ET = np.eye(q)[:,:p]
        B = ET@np.linalg.inv(EQE)@EQ
        IB = np.eye(q) - B
        L = IB@np.linalg.cholesky(Q)
        return L.T