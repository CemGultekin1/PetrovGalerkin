from typing import List
from solver.dimensional import Dimensional
from .bndrcond import BoundaryCondition
from .matalloc import BlockColumns, TriRowColumn
from .boundary import BoundaryElementFactory
from .interior import InteriorElementFactory
from chebyshev import ChebyshevInterval
import numpy as np
class EquationFactory(Dimensional):
    def __init__(self,max_degree:int,boundary_condition:BoundaryCondition = BoundaryCondition(np.empty((0,0)),np.empty((0,0)),np.empty(0,)),) -> None:        
        super().__init__()
        self.max_degree = max_degree
        self.interr = InteriorElementFactory(max_degree)
        self.interr.fillup()
        self.bndr = BoundaryElementFactory(max_degree)  
        self.bndr.fillup()
        self.bndr_cond = self.bndr.create_boundary_condition_element_factory(boundary_condition)
        self.setup_handle :List[EquationFactory] = []
    def setup_for_operations(self,boundary_condition:BoundaryCondition):
        self.bndr_cond = self.bndr.create_boundary_condition_element_factory(boundary_condition)
        for eq in self.setup_handle:
            eq.setup_for_operations(boundary_condition)
    def change_boundary_condition(self,bcond:BoundaryCondition)->'EquationFactory':
        lef = EquationFactory.__new__(EquationFactory)
        lef.__dict__.update(self.__dict__)
        lef.bndr_cond = self.bndr.create_boundary_condition_element_factory(bcond)
        return lef
    def generate_interior_matrices(self,center_chebint:ChebyshevInterval,\
                            left_degree:int,right_degree:int,target_degree:int,):

        interrelem = self.generate_local_quadratures(center_chebint,target_degree,).to_matrix_form()
        bndrelem = self.bndr.generate_element(left_degree,target_degree,right_degree,).to_matrix_form(self.dim)
        return BlockColumns(interrelem,bndrelem)
    def generate_quad_interior_time_derivatives_element(self,center_chebint:ChebyshevInterval,target_degree:int,left :bool= False):
        matfun,rhsfun = center_chebint.separate_funs()
        mat_rhs = self.interr.generate_quad_interior_time_derivatives_element(target_degree,matfun,rhsfun,left = left).mat_rhs_matrices()
        return mat_rhs
    def generate_quad_boundary_time_derivatives_element(self,center_chebint:ChebyshevInterval,target_degree:int,left:bool = True):
        matfun,rhsfun = center_chebint.separate_funs()
        mat_rhs = self.interr.generate_quad_boundry_time_derivatives_element(target_degree,matfun,rhsfun,left = left).mat_rhs_matrices()       
        return  mat_rhs
    def generate_local_quadratures(self,center_chebint:ChebyshevInterval,target_degree:int,):
        matfun,rhsfun = center_chebint.separate_funs()
        interrelem = self.interr.generate_element(target_degree,matfun,rhsfun,)
        return interrelem
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




