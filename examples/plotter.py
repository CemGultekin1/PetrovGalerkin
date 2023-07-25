
import logging
from typing import Tuple, Union
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
    def __init__(self,per_interval_pts_num :int = 64,flux_sep:float = 0.1) -> None:
        self.per_interval_pts_num = per_interval_pts_num
        self.flux_sep = flux_sep
    def draw(self,chebint:GridwiseChebyshev,fig:plt.Figure = None,axs:plt.Axes = None)->Tuple[plt.Figure,plt.Axes]:
        if axs is None:
            fig,axs = plt.subplots(1,1,figsize = (8,5))
        sep = np.amin(chebint.hs)
        fs = self.flux_sep
        markersize = 5
        mmc = MinMaxCollector()
        # def xtransform():
            
        for i,cheb in enumerate(chebint.cheblist):
            a,b = cheb.interval
            x = np.linspace(a,b,self.per_interval_pts_num)
            y = cheb(x)
            x_ = np.linspace(i,i+1,self.per_interval_pts_num)
            a_,b_ = x_[0],x_[-1]
            axs.plot(x_,y,color = 'b')
            yl = cheb.left_value()
            yr = cheb.right_value()
            mmc.enter(y)
            mmc.enter([yl,yr])
            
            axs.plot([a_+fs,b_-fs],[yl,yr],'r.',markersize = markersize)
        axs1 = axs.twinx()
        for i,cheb in enumerate(chebint.cheblist):
            x_ = np.linspace(i,i+1,2)
            a_,b_ = x_[0],x_[-1]
            x = np.linspace(a_,b_,2)
            axs1.bar(np.mean(x),cheb.degree,b_-a_,color = 'g',alpha = 0.5)
        yl = chebint.edge_values.get_interval_edge(0,left = True,)
        yr = chebint.edge_values.get_interval_edge(-1,right = True,)
        mmc.enter([yl,yr])
        axs.plot([-fs,b_+fs],[yl,yr],'r.',markersize = markersize)
        exts  = mmc.extend_extremums()
        edges = np.array(chebint.edges)
        for i,edge in enumerate(edges):
            axs.plot([i,i], exts,'k--',linewidth= 1)
        axs.set_ylim(exts)
        n = len(edges) - 1
        ticks = np.unique(np.linspace(0,n,8).astype(int))
        axs.set_xticks(ticks)
        tlabels = ["{:.1e}".format(t) for t in edges[ticks]]
        axs.set_xticklabels(tlabels,rotation = -25)
        return fig,axs
class MultiDimGridwiseChebyshevPlotter(GridwiseChebyshevPlotter):
    def __init__(self, per_interval_pts_num: int = 64, flux_sep: float = 0.05,) -> None:
        super().__init__(per_interval_pts_num, flux_sep)
    def draw(self, chebint: GridwiseChebyshev):
        dim = chebint.dim
        for i in range(dim):
            yield super().draw(chebint.separate_dims(dim,index = i))