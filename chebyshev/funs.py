from typing import Callable, Dict, Tuple, Union
import numpy as np

NumericType= Union[float,np.ndarray]
NumericFunType = Callable[[NumericType],NumericType]

class ListOfFuns(list):
    def __init__(self,*funs:NumericFunType):
        list.__init__(self,)
        self.extend(funs)
    def __getitem__(self,i:int)->NumericFunType:
        return super().__getitem__(i)
    def eval_rule(self,fun,x):
        y = fun(x)
        return y
    def __call__(self,x:NumericType)->Tuple[NumericType]:
        outs = []
        for fun in self:
            outs.append(self.eval_rule(fun,x))
        return tuple(outs)
    def flatten(self,):
        return FlatListOfFuns(*self)
class FlatListOfFuns(ListOfFuns):
    def eval_rule(self, fun, x):
        val =  super().eval_rule(fun,x)
        val.reshape([len(x),-1])
        return val        
    def __call__(self, x: NumericType) -> NumericType:
        vals = super().__call__(x)
        vals = tuple(val.reshape(len(x),-1) for val in vals)
        return np.concatenate(vals,axis = 1)
    def shapen(self,):
        return ListOfFuns(*self)