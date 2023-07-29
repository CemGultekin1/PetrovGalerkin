

class Dimensional:
    dim:int
    def __init__(self,dim:int = 1) -> None:
        self.dim = dim
    def set_dim(self,dim:int):
        self.dim = dim
        for x,y in self.__dict__.items():
            if issubclass(y.__class__,Dimensional):
                y.set_dim(dim)