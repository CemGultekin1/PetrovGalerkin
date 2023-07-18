
import logging
from typing import Union
from chebyshev.interval import GridwiseChebyshev
import numpy as np
import matplotlib.pyplot as plt

class MinMaxCollector:
    def __init__(self) -> None:
        self.min,self.max = np.inf,-np.inf
    def enter(self,val:Union[float,np.ndarray,int]):
        if np.isscalar(val):
            if val < self.min:
                self.min = val
            if val > self.max:
                self.max = val
        else:
            val_ = np.amax(val)
            self.enter(val_)
            val_ = np.amin(val)
            self.enter(val_)
    def give_extremums(self,):
        return np.array([self.min,self.max])
    def extend_extremums(self,ratio = 0.05):
        ext = self.give_extremums()
        sp =np.mean(np.abs(ext))*ratio
        if sp == 0:
            sp = 0.1
        ext[0] -= sp
        ext[1] += sp
        return ext
class GridwiseChebyshevPlotter:
    def __init__(self,per_interval_pts_num :int = 64,flux_sep:float = 0.05) -> None:
        self.per_interval_pts_num = per_interval_pts_num
        self.flux_sep = flux_sep
    def draw(self,chebint:GridwiseChebyshev):
        fig,axs = plt.subplots(1,1,figsize = (5,5))
        sep = np.amin(chebint.hs)
        fs = sep*self.flux_sep
        markersize = 5
        mmc = MinMaxCollector()
        for cheb in chebint.cheblist:
            a,b = cheb.interval
            x = np.linspace(a,b,self.per_interval_pts_num)
            y = cheb(x)
            axs.plot(x,y,color = 'b')
            yl = cheb.left_value()
            yr = cheb.right_value()
            mmc.enter(y)
            mmc.enter([yl,yr])
            axs.plot([a+fs,b-fs],[yl,yr],'r.',markersize = markersize)
        yl = chebint.edge_values.get_interval_edge(0,left = True,)
        yr = chebint.edge_values.get_interval_edge(-1,right = True,)
        mmc.enter([yl,yr])
        axs.plot([-fs,b+fs],[yl,yr],'r.',markersize = markersize)
        exts  = mmc.extend_extremums()
        edges = chebint.edges
        for edge in edges:
            axs.plot([edge,edge], exts,'k--',linewidth= 1)
        axs.set_ylim(exts)
        return fig