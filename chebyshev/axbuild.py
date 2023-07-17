from typing import Tuple, Union
import numpy as np
class Axis:
    def __init__(self,size:int,name:str = '') -> None:
        self.size = size
        self.name = name
class MergedAxis(Axis):
    def __init__(self,*axes:Axis,name:str = '') -> None:
        self.axes = axes
        self.size = np.prod(tuple(ax.size for ax in axes))  
        self.name = name
    def to_axis(self,):
        return Axis(self.size,name = self.name)
class MergedAxes(Axis,list):
    def __init__(self,*merged_axes:MergedAxis,name:str = '') -> None:
        list.__init__(self,)
        if merged_axes:
            sz = 1
            for ma in merged_axes:
                self.append(ma)
                sz *= ma.size            
        else:
            sz = 0            
        Axis.__init__(self,sz, name=name)                    
    def __getitem__(self,i:Union[int,str])->MergedAxis:
        if isinstance(i,str):
            for ii in range(len(self)):
                ma :MergedAxis= super().__getitem__(ii)
                if ma.name == i:
                    return ma
            return None
        return super().__getitem__[i]        
class Coords:
    delimeter:str = '-'
    def __init__(self,*axes:Axis) -> None:
        self.axes = axes
        self.shape = tuple(ax.size for ax in axes)
        self.size = sum(self.shape)
        self.names = [ax.name for ax in self.axes]
    @classmethod
    def from_tuple(*tup:Tuple[int,str]):
        return Coords(*tuple(Axis(*tup_) for tup_ in tup))
    @property
    def dim(self,):
        return len(self.axes)
    def reshape(self,arr:np.ndarray):
        return arr.reshape(self.shape)
    @classmethod
    def from_merged_axes(cls,mergedax:MergedAxes)->'Coords':
        axes = tuple(maxis.to_axis() for maxis in mergedax)
        return Coords(*axes)
    def ax_index(self,name:str):
        return self.names.index(name)
    def to_merged_axes(self,name:str = ''):
        if not name:
            name = self.default_coord_name
        return MergedAxes(*tuple(MergedAxis((ax,),name = ax.name) for ax in self.axes),name = name)
    @property
    def default_coord_name(self,):
        return self.delimeter.join(self.names)        
class CoordinateTransformation:
    def __init__(self,outcoords:Union[MergedAxes,Coords]) -> None:
        if isinstance(outcoords,MergedAxes):
            outcoords = Coords.from_merged_axes(outcoords)
        self.outcoords = outcoords
    def transform(self,arr:np.ndarray)->np.ndarray:...    

class Reshaping(CoordinateTransformation):
    def __init__(self, out_axes: Union[MergedAxes,Coords]) -> None:        
        super().__init__(out_axes)        
        self.shape = self.outcoords.shape
    def transform(self,arr:np.ndarray)->np.ndarray:
        arr = arr.reshape(self.shape)
        return arr    
class OuterProduct(CoordinateTransformation):
    incoords:Tuple[Coords]
    def __init__(self, *incoords: Coords, ) -> None:
        self.incoords = incoords
        axes = []
        for coords in incoords:
            axes.extend(coords.axes)
        super().__init__(Coords(*axes))
        self.ncoords = len(self.incoords)
        self.product_shape,self.final_shape = self.build_reshape_tuples()
    def build_reshape_tuples(self,):
        pshp = [[1]*self.ncoords]*self.ncoords
        for  i in range(self.ncoords):
            pshp[i][i] = self.incoords[i].size
        fshp = []
        for coords in self.incoords:
            fshp.extend(coords.shape)
        return pshp,fshp
    def transform(self, *arrs: np.ndarray) -> np.ndarray:
        for i, (arr,shp) in enumerate(zip(arrs,self.product_shape)):
            arrs[i] = arr.reshape(shp)
        arr = arrs[0]
        for i in range(1,len(arrs)):
            arr = np.multiply(arr,arrs[i])
        arr = arr.reshape(self.final_shape)
        return arr
class TranspositionCondition(CoordinateTransformation):
    def __init__(self, incoords: Coords, merge_axes: MergedAxes) -> None:
        super().__init__(merge_axes,)
        self.indexes = list(list(incoords.ax_index(ax.name) for ax in mergax.axes) for mergax in merge_axes)        
        self.flat_indexes = list(ind__ for ind_ in self.indexes for ind__ in ind_) 
        self.flat_length = len(self.flat_indexes)
        self.transposition_needed = self.flat_indexes == tuple(range(self.flat_length))
        assert len(np.unique(self.flat_indexes)) == self.flat_length
    def is_transposition_needed(self,):
        return self.flat_indexes == range(len(self.flat_indexes))
class Transposition(TranspositionCondition):
    def __init__(self, incoords: Coords, merge_axes: MergedAxes,conditional:bool = False) -> None:
        super().__init__(incoords,merge_axes)
        self.conditional = conditional
    def transform(self,arr:np.ndarray)->np.ndarray:
        if self.conditional and self.transposition_needed:
            arr = arr.transpose(self.flat_indexes)
        return arr
class CoordinateTransformations(CoordinateTransformation):  
    def transform(self, arr: np.ndarray) -> np.ndarray:
        for val in self.__dict__.values():
            if isinstance(val,CoordinateTransformation):
                arr = val.transform(arr)
        return arr
class FitCoordinate(CoordinateTransformations):
    def __init__(self, incoords: Coords, merge_axes: MergedAxes) -> None:
        super().__init__(merge_axes)
        self.reshape1 = Reshaping(incoords,)        
        self.transposition = Transposition(incoords, merge_axes, conditional=True)
        self.reshape2 = Reshaping(merge_axes,)
class PreMatMultip(Reshaping):
    def __init__(self, incoords: Coords,from_left:bool = False,from_right:bool = False) -> None:
        self.incoords = incoords
        assert from_left + from_right == 1
        shp = self.incoords
        self.from_left = from_left
        if from_left:
            self.shape = (shp[0],sum(shp[1:]))
        else:
            self.shape = (sum(shp[:-1]),shp[-1])
    def transform(self, arr: np.ndarray,arr2:np.ndarray) -> np.ndarray:
        if self.from_left:
            arr = super().transform(arr)
        else:
            arr2 = super().transform(arr2)
        return arr,arr2
    
    
class SimpleMatMultip(CoordinateTransformation):
    def __init__(self, left_coords: Coords,right_coords:Coords) -> None:
        axes = list(left_coords.axes[:-1]) + list(right_coords.axes[1:])
        super().__init__(Coords(*axes))
    def transform(self, arrleft: np.ndarray,arrright:np.ndarray) -> np.ndarray:
        return arrleft @ arrright 
    
       
class MultiMatMultip(CoordinateTransformations):
    def __init__(self, leftmat: Coords, rightmat: Coords) -> None:
        assert leftmat.axes[-1].name == rightmat.axes[0].name
        leftmat.shape
        self.reshape1 = PreMatMultip(rightmat,from_left = True)
        self.reshape2 = PreMatMultip(leftmat,from_right = True)
        self.matmult = SimpleMatMultip(leftmat,rightmat)
        self.reshape3 = Reshaping(self.matmult.outcoords)
