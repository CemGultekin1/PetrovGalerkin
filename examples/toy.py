import itertools
import numpy as np


def valuemat():
    return np.array([[1,1],[-1,1]])
def integmat():
    return np.array([[0,0],[2,0]])


def tricol(ravg:float= 1/2,lavg:float= 1/2):
    vals = valuemat()
    ints = integmat()
    center = np.zeros((2,2))
    left = np.zeros((2,2))
    right = np.zeros((2,2))
    for i,j in itertools.product(*[range(2)]*2):
        vu = vals[i,1]*vals[j,1]*ravg - vals[i,0]*vals[j,0]*lavg
        dvu =  ints[i,j]
        print(f'center[{i,j}] = vu - dvu = {vu} - {dvu}')
        center[i,j] = vu - dvu
    
        left[i,j] = vals[i,1]*vals[j,0]*lavg
        right[i,j] = -vals[i,0]*vals[j,1]*ravg
    return center, right,left
def build_tricol(**kwargs):
    c,r,l = tricol(**kwargs)
    return c,r,l
def get_slices(c,i,j,mult,offsets):
    return tuple(slice(mult*ind+off,mult*ind+c_+off) for c_,ind,off in zip(c.shape,(i,j),offsets))
def place_block(mat,block,i,j,mult = 2,offsets = (0,0)):
    slc = get_slices(block,i,j,mult = mult,offsets = offsets)
    mat[slc] = block
    return mat

def build_global(n:int,avgs):
    k1 = n*2
    k2 = (n+2)*2
    x = np.zeros((k2,k1))
    kwargs = dict(mult = 2,)
    #interior
    for i in range(n):
        lavg,ravg = avgs[i,0],avgs[i,1]        
        c,r,l = build_tricol(lavg = lavg,ravg = ravg)
        x = place_block(x,r,i+2,i,**kwargs)
        x = place_block(x,c,i+1,i,**kwargs)
        x = place_block(x,l,i,i,**kwargs)
    return x[2:-2,:]
def build_rhs(n):
    rhs = np.zeros((2*n))
    rhs[0] = 1/n
    rhs[1] = 0
    for i in range(1,n):
        rhs[2*i] = 1/n
        rhs[2*i+1] = 0
    
    return rhs
def add_axiliary_rhs(rhs,c):
    return np.concatenate([rhs,[c,],[c,]],axis = 0)
def add_auxiliaries(x,avgs):
    vals = valuemat()
    z = np.zeros(x.shape[0] - vals.shape[0])
   
    left = np.concatenate([-vals[:,0],z],axis = 0).reshape([-1,1])*(1 - avgs[0,0])
    right = np.concatenate([z,vals[:,1]],axis = 0).reshape([-1,1])*(1 - avgs[-1,1])
    x = np.concatenate([left,x,right],axis = 1)
    return x
def add_auxiliary_boundary(x,b0,b1):
    vals = valuemat()
    
    z0 = np.zeros(x.shape[1])
    left = np.concatenate([[0],vals[:,0]],axis = 0)*b0
    right = np.concatenate([vals[:,1]*0,[1]],axis = 0)*b1
    z0[:len(left)] += left
    z0[-len(right):] += right
    
    z1 = np.zeros(x.shape[1])
    left = np.concatenate([[1],vals[:,0]*0],axis = 0)*b0
    right = np.concatenate([vals[:,1],[0]],axis = 0)*b1
    z1[:len(left)] += left
    z1[-len(right):] += right
    
    
    
    z0,z1 = (z.reshape([1,-1]) for z in (z0,z1))
    return np.concatenate([x,z1,z0,],axis = 0)
def build_avgs(n):
    avgs = np.ones((n,2))/2
    # avgs[0,0] = 0
    # avgs[-1,1] = 1
    
    # avgs[0,1] + avgs[1,0]
    
    assert np.all((avgs[:-1,1] + avgs[1:,0])==1)
    return avgs
def main():
    n = 1
    b0,b1,c = 1,0,0
    avgs = build_avgs(n)
    x = build_global(n,avgs)
    x = add_auxiliaries(x,avgs)
    x = add_auxiliary_boundary(x,b0,b1)
    
    rhs = build_rhs(n)
    rhs = add_axiliary_rhs(rhs,c)
    
    print(x)
    print(rhs)

   

    tr_left_val = (c - b1)/(b0 + b1)
    tr_right_val = tr_left_val + 1
    print(f'left_val\' = {tr_left_val}, right_val\' = {tr_right_val}')
    
    sol = np.linalg.solve(x,rhs)
    
    valmat = valuemat()
    
    left_val = (sol[0] + sol[1:3]@valmat[:,0])/2
    right_val = (sol[-1] + sol[-3:-1]@valmat[:,1])/2
    print(f'left_val  = {left_val}, right_val  = {right_val}')
    
    print(sol)
if __name__ == '__main__':
    main()
