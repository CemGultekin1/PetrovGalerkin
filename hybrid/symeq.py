import sympy as sym
from sympy.utilities import lambdify
import numpy as np
vnames = 'm0s, invt1, invt2f, r, t2s, theta1, theta2, rfpulse, invtrf, reltime'
m0s,invt1,invt2f,r,t2s,theta1,theta2,rfpulse,invtrf,reltime = sym.symbols(vnames)

vnames = vnames.split(', ')

invtrf_ = 2000
alpha = theta2 + theta1
theta = theta1*(1 - rfpulse) + ((theta2 + theta1)*reltime*invtrf_ - theta1)*rfpulse
dtheta = alpha*rfpulse*invtrf_
mat = sym.Matrix(
    [[sym.cos(theta)**2*invt1 - sym.sin(theta)**2*invt2f - sym.cos(theta)**2*r*m0s,
        r*(1-m0s)*sym.cos(theta)],[
            r*m0s*sym.cos(theta),
            -invt1 - t2s*dtheta**2]]
)

rhs = sym.Matrix( [(1-m0s)*sym.cos(theta)*invt1,m0s*invt1])
bndr = sym.Matrix([
    [-1/(1- sym.pi**2*t2s/1e-3),0],
        [0,1/(1- sym.pi**2*t2s/1e-3)]
])
zmat = sym.Matrix(
    [[0,0],[0,0]]
)

param_vars = [m0s,invt1,invt2f,r,t2s]
design_vars = [theta1,theta2,invtrf]
extra_vars = [reltime,rfpulse]

all_vars = param_vars + design_vars + extra_vars

mats = [mat]
rhss = [rhs]
bndrs = [bndr]
for pv in param_vars:
    dmat = mat.diff(pv)*pv
    mats.append(dmat)
    drhs = rhs.diff(pv)*pv
    rhss.append(drhs)
    dbndr = bndr.diff(pv)*pv
    bndrs.append(dbndr)

form_mat = sym.Matrix.vstack(*mats)
form_rhs = sym.Matrix.vstack(*rhss)
form_bndr = sym.Matrix.vstack(*bndrs)
zmats = [zmat]*len(mats)
for i in range(1,len(mats)):
    zmats[i] = mat
    zcol = sym.Matrix.vstack(*zmats)
    form_mat = sym.Matrix.hstack(form_mat,zcol)
    zmats[i] = zmat
    
for i in range(1,len(mats)):
    zmats[i] = bndr
    zcol = sym.Matrix.vstack(*zmats)
    form_bndr = sym.Matrix.hstack(form_bndr,zcol)
    zmats[i] = zmat

form_rhs_design = {}
form_mat_design = {}
form_bndr_design = {}
for dv in design_vars:
    form_mat_design[dv.name] = form_mat.diff(dv)
    form_rhs_design[dv.name] = form_rhs.diff(dv)
    form_bndr_design[dv.name] = form_bndr.diff(dv)


org_sys_mats = (mat,rhs,bndr)
params_sys_mats = (form_mat,form_rhs,form_bndr)
design_sys_mats = (form_mat_design,form_rhs_design,form_bndr_design)


org_sys_mats = tuple(lambdify(all_vars,x) for x in org_sys_mats)
params_sys_mats = tuple(lambdify(all_vars,x) for x in params_sys_mats)
design_sys_mats = tuple({key:lambdify(all_vars,y) for key,y in x.items()} for x in design_sys_mats)

def subsdict(*args):
    return {vn:arg for vn,arg in zip(vnames,args)}

class MatSelect:
    def __init__(self,selnum) -> None:
        self.org_flag = selnum == 0
        self.params_flag = selnum == 1
        self.design_flag = selnum == 2
    def type_select(self,):
        if self.org_flag:
            return org_sys_mats
        elif self.params_flag:
            return params_sys_mats
        elif self.design_flag:
            return design_sys_mats
    def subtype_select(self,mat_flag = False,rhs_flag = False,bndr_flag = False):
        i = mat_flag*0 + rhs_flag*1 + bndr_flag*2
        assert mat_flag + rhs_flag + bndr_flag == 1
        return self.type_select()[i]
    def __call__(self,fun):
        def wrapped_fun(*args,mat_flag = False,rhs_flag = False,bndr_flag = False,**kwargs):
            sys = self.subtype_select(mat_flag,rhs_flag,bndr_flag)
            val =  fun(sys,*args,**kwargs)
            val = np.array(val).astype(np.float64).squeeze()
            return val
        return wrapped_fun
    
@MatSelect(0)
def org_sys(ssmat,*args,):
    return ssmat(*args)#.evalf(subs = subsdict(*args))

@MatSelect(1)
def params_sys(ssmat,*args,):
    return ssmat(*args)#.evalf(subs = subsdict(*args))

@MatSelect(2)
def design_sys(ssmat,*args,name: str = 'theta1',):
    return ssmat[name](*args)#.evalf(subs = subsdict(*args))


# def main():
#     mat = design_sys(0.1,0.1,0.1,0.1,0.1,0.1,0.1,bndr_flag = True,name = 'theta')
#     print(mat.shape)

# if __name__ == '__main__':
#     main()