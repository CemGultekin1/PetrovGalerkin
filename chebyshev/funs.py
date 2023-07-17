from typing import Callable, Dict, Tuple, Union
import numpy as np

NumericType= Union[float,np.ndarray]
NumericFunType = Callable[[NumericType],NumericType]
class ShapeRecognition:
    def __init__(self) -> None:
        self.tuple = ()
        self.is_recognized = False
    def recognize(self,shp:Tuple[int,...]):
        if self.is_recognized:
            return
        self.tuple = shp
        self.is_recognized = True
        
class ConcatenatedVectorSeparator:
    def __init__(self,*shps:Tuple[int,...]) -> None:
        self.shps = shps
        self.indices = np.cumsum([np.prod(shp) for shp in shps])
    def reshape_collapsed_axis(self,vec:np.ndarray,axis = 1)->Tuple[np.ndarray,...]:
        vecs =  np.split(vec,self.indices[:-1],axis = axis)
        shps = tuple(list(vec_.shape) for vec_ in vecs)
        shps = tuple(list(shp_[:axis]) + list(shp) + list(shp_[axis+1:]) for shp_,shp in zip(shps,self.shps))
        vecs = tuple(vec.reshape(shp) for vec,shp in zip(vecs,shps))
        return vecs
class FunShapes:
    def __init__(self,nfuns:int) -> None:
        self.separator = None
        self.shapes_list = [ShapeRecognition() for _ in range(nfuns)]
        self.all_recognized = False
    def check_all_recognitions(self,):
        for shp in self.shapes_list:
            if not shp.is_recognized:
                return False
        return True
    def init_separator(self,):
        shps = [shp.tuple for shp in self.shapes_list]
        self.separator =ConcatenatedVectorSeparator(*shps)
    def recognize_shape(self,val:np.ndarray,i:int,x:NumericType):
        if self.all_recognized:
            return
        sr = self.shapes_list[i]
        if sr.is_recognized:
            return
        if np.isscalar(x):
            sr.recognize(val.shape)
        else:
            shp = val.shape[x.ndim:]
            sr.recognize(shp)
        self.all_recognized = self.check_all_recognitions()
        if self.all_recognized:
            self.init_separator()
class ListOfFuns(list,FunShapes):
    def __init__(self,*funs:NumericFunType):
        FunShapes.__init__(self,len(funs))
        list.__init__(self,)        
        self.extend(funs)
        
    def __getitem__(self,i:int)->NumericFunType:
        return super().__getitem__(i)
    def eval_rule(self,fun,x,i):        
        y = fun(x)
        self.recognize_shape(y,i,x)
        return y
    def __call__(self,x:NumericType)->Tuple[NumericType]:
        outs = []
        for i,fun in enumerate(self):
            outs.append(self.eval_rule(fun,x,i))
        return tuple(outs)
    def flatten(self,):
        return FlatListOfFuns(*self)
class FlatListOfFuns(ListOfFuns):
    def eval_rule(self, fun, x,i):
        val =  super().eval_rule(fun,x,i)
        val.reshape([len(x),-1])
        return val        
    def __call__(self, x: NumericType) -> NumericType:
        vals = super().__call__(x)
        vals = tuple(val.reshape(len(x),-1) for val in vals)
        return np.concatenate(vals,axis = 1)
    def shapen(self,):
        return ListOfFuns(*self)
    def separate_vector(self,vec:np.ndarray,axis :int = 1):
        return self.separator.reshape_collapsed_axis(vec,axis = axis)
        