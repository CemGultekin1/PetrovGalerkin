from datetime  import datetime
import os
from fldrsys.defaults import OUTPUTS

class Folder:
    path:str
    def __init__(self,*conns) -> None:
        self.path =os.path.join(*conns)
    def create(self,):
        if not os.path.exists(self.path):
            os.makedirs(self.path)     
        return self  
    def to_str(self,):
        return str(self)
    def __str__(self,):
        return self.path
    
class ExampleFolderNames:
    def __init__(self,root:str,format = "%Y-%m-%d") -> None:
        self.format = format
        self.root = root
        self.utc_str = datetime.utcnow().strftime(format)    
    def from_file_name(self,flname:str):
        flname = flname.split('.')[0]
        flname = flname.split('/')[-1]
        return self.new_folder(flname)
    def new_folder(self,name:str):
        return Folder(self.root,self.utc_str,name)
    
class OutputsFolders(ExampleFolderNames):
    def __init__(self,foldername:str = '',format:str = "%Y-%m-%d") -> None:
        if not bool(foldername):
            foldername = OUTPUTS
        super().__init__(foldername,format = format)
        