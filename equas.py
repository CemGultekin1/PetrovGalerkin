import itertools
from chebyshev.interval import ChebyshevFactory, ChebyshevInterval, GridwiseChebyshev
from chebyshev.interior import InteriorElements, InteriorElementFactory, QuadratureTensor
from chebyshev.boundary import BoundaryElementFactory, BoundaryElements, QuadraticBoundary
import numpy as np
# class TestEquation(ChebyshevFactory):

# class GlobalEquation:
#     def __init__(self,matfun:GridwiseChebyshev,rhs:GridwiseChebyshev):
#         super().__init__()
#         self.matfun = matfun
#         self.rhs = rhs

class GlobalElement(InteriorElements,BoundaryElements):
    def __init__(self,intel:InteriorElements,bdel: BoundaryElements) -> None:
        assert intel.dimensionalized == bdel.dimensionalized
        self.__dict__.update(intel.__dict__)
        self.__dict__.update(bdel.__dict__)
        
class GlobalElementFactory(ChebyshevFactory):
    def __init__(self) -> None:
        bd_fact = BoundaryElementFactory()
        bd_fact.fillup()
        self._bd_el = bd_fact.create_elements()
        self.in_fact = InteriorElementFactory()
        self.in_fact.fillup()
    def __call__(self,matfun:ChebyshevInterval,rhsfun:ChebyshevInterval):
        in_el = self.in_fact.generate_elements(matfun,rhsfun)
        return GlobalElement(in_el,self._bd_el)
    
# class GlobalLinearSystem:
    
        
    

def linear_system_entry(matfun:ChebyshevInterval,rhs:ChebyshevInterval,qtensor:QuadratureTensor):
    matc = matfun.coeffs
    rhsc = rhs.coeffs
    mate = qtensor.tri_multip(matc)
    rhse = qtensor.dub_multip(rhsc)
    return mate,rhse
        
        