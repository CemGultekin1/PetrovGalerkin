import numpy as np
class MatPrinter:
    def __init__(self,width:int,decimals:int) -> None:
        self.formatter = '{' + f':{width}.{decimals}f'+ '}'
    def to_str(self,mat:np.ndarray):
        rows = []
        for i in range(mat.shape[0]):
            row = mat[i]
            if np.isscalar(row):
                rowstr = self.formatter.format(row)
            else:
                rowstr = ','.join([self.formatter.format(x) for x in row])
            rows.append(rowstr)
        rows = '\n'.join(rows)
        return rows
    