

from typing import List





class Parameteric:
    params_list: List[str] = []
    def change_of_parameters(self,**kwargs):
        for attr in self.params_list:
            self.__dict__[attr] = kwargs.get(attr,self.__dict__[attr])
        for x,y in self.__dict__.items():
            if issubclass(y.__class__,Parameteric):
                y.change_of_parameters(**kwargs)


class DesignParameteric:
    design_params_list: List[str] = []
    def change_of_design_parameters(self, **kwargs):
        for attr in self.design_params_list:
            self.__dict__[attr] = kwargs.get(attr,self.__dict__[attr])
        for x,y in self.__dict__.items():
            if issubclass(y.__class__,Parameteric):
                y.change_of_design_parameters(**kwargs)
                
