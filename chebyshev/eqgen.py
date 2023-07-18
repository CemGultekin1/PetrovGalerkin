from chebyshev.core import BoundaryCondition
from chebyshev.matalloc import BlockColumns, TriRowColumn
from .boundary import BoundaryElementFactory
from .interior import InteriorElementFactory
from .interval import ChebyshevInterval
from .funs import FlatListOfFuns


class LocalEquationFactory:
    def __init__(self,dim:int,max_degree:int,boundary_condition:BoundaryCondition) -> None:        
        self.dim = dim
        self.interr = InteriorElementFactory(max_degree)
        self.interr.fillup()
        self.bndr = BoundaryElementFactory(max_degree)        
        self.bndr.fillup()
        self.bndr_cond = self.bndr.create_boundary_condition_element_factory(boundary_condition)
        
    def generate_interior_matrices(self,funs:FlatListOfFuns,\
                    center_chebint:ChebyshevInterval,\
                            left_degree:int,right_degree:int,):
        center_degree = center_chebint.degree
        matfun,rhsfun = center_chebint.separate_funs(funs)
        interrelem = self.interr.generate_element(matfun,rhsfun,).to_matrix_form()
        bndrelem = self.bndr.generate_element(left_degree,center_degree,right_degree,).to_matrix_form(self.dim)
        return BlockColumns(interrelem,bndrelem)
    
    def generate_boundary_blocks(self,center_degree:int,\
                            left_degree:int,right_degree:int,):
        bndrelem = self.bndr.generate_element(left_degree,center_degree,right_degree,).to_matrix_form(self.dim)
        return TriRowColumn(bndrelem.mat_left,bndrelem.mat_center,bndrelem.mat_right,None)
            
    def generate_boundary_condition_matrices(self,leftmost_degree:int,rightmost_degree:int):
        bce = self.bndr_cond.generate_element(leftmost_degree,rightmost_degree)
        bcemat = bce.to_matrix_form()
        mat0 = TriRowColumn(None,bcemat.mat_b0,None,bcemat.rhs_c)
        mat1 = TriRowColumn(None,bcemat.mat_b1,None,None)        
        return mat0,mat1



