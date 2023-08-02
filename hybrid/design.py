import numpy as np
from chebyshev import GridwiseChebyshev
class InvTrfFunction:
    def __init__(self,trf_seq:np.ndarray,tr:float) -> None:
        self.tr = tr
        self.trf_seq = trf_seq
        n = len(self.trf_seq)
        self.fingerprint_edges = np.arange(1,n+1)*tr
    def give_trf_edges(self,i:int,):
        return self.tr*(i+1) - self.trf_seq[i]/2,self.tr*(i+1) + self.trf_seq[i]/2
    def time_derivatives_to_invtrf(self,dldinst:np.ndarray,solution:GridwiseChebyshev):
        dldinvtrf = np.zeros(self.trf_seq.shape)
        invtrf = 1/self.trf_seq
        for i in range(len(self.trf_seq)):
            left,right = self.give_trf_edges(i)
            ileft,iright = solution.find_closest_edges((left,right))
            # left time instance t = i*tr - trf/2
            # dt/dtrf = -1/2
            # dt/dinvtrf = dt/dtrf * dtrf/dinvtrf = (-1/2)*(-1/invtrf**2) = 1/(2*invtrf**2)
            ileft -= 1
            iright -= 1
            invtrfi = invtrf[i]
            dldinvtrf[i] = dldinst[ileft]/(2*invtrfi**2) - dldinst[iright]/(2*invtrfi**2)
        return dldinvtrf
    def rf_pulse_to_invtrf(self,dldrfpulse:np.ndarray,solution:GridwiseChebyshev):
        dldinvtrf = np.zeros(self.trf_seq.shape)
        invtrf = 1/self.trf_seq
        for i in range(len(self.trf_seq)):
            left,right = self.give_trf_edges(i)
            ileft,iright = solution.find_closest_edges((left,right))
            dldinvtrf[i] = np.sum(dldrfpulse[ileft:iright])
        return dldinvtrf
                