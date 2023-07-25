import logging
from typing import Tuple
from .quadrature import InnerProducts
import numpy as np
from .funs import NumericFunType
from .interval import coeffgen
class ResidualNorm(InnerProducts):
    def __init__(self, degree: int) -> None:
        super().__init__(2*degree)
        middegree = 1
        self.orthogonals_projections = {deg:HalfOrderOrthogonalProjection(deg,self) for deg in range(middegree,degree+1)}
        self.orthogonalization_cholesky = {deg:np.empty(0) for deg in range(middegree,degree+1)}
    def fillup(self):
        super().fillup()
        self.orthogonalization_cholesky = {deg:op.compute() for deg,op in self.orthogonals_projections.items()}    
    def residual_norm(self,coeffs:np.ndarray,)->float:
        deg = coeffs.shape[0]
        coeffs = coeffs.reshape([deg,-1])
        hnorm = self.orthogonalization_cholesky[deg//2]@coeffs
        return np.sum(hnorm**2,)
        # return np.linalg.norm(hnorm,'fro')
    def residual_norm_from_fun(self,fun:NumericFunType,interval:Tuple[int,int],degree:int):
        '''
        takes 2*degree chebyshev transformation
        computes the residual norm for best fit degree
        '''
        if degree > self.degree//2:
            logging.error(f'Requested degree is not supported! {degree} > {self.degree//2} ')
            raise Exception
        coeffs = coeffgen(fun,2*degree-1,outbounds = interval,)
        return self.residual_norm(coeffs)
        
class HalfOrderOrthogonalProjection:
    def __init__(self,half_degree:int,innprod:InnerProducts) -> None:
        self.half_degree = half_degree
        self.innprod = innprod
    def compute(self,):
        p = self.half_degree
        q = self.half_degree*2
        Q = self.innprod.dub_quads[:q,:q]
        EQE = Q[:p,:p]
        EQ = Q[:p,:]
        ET = np.eye(q)[:,:p]
        B = ET@np.linalg.inv(EQE)@EQ
        IB = np.eye(q) - B
        L = IB@np.linalg.cholesky(Q)
        return L.T

