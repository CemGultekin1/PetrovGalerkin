import itertools
import numpy as np


def valuemat():
    return np.array([[1,1],[-1,1]])
def integmat():
    return np.array([[0,0],[2,0]])


def tricol(left_edge:bool = False, right_edge:bool = False):
    print(f'left_edge = {left_edge},\t right_edge = {right_edge} ')
    vals = valuemat()
    ints = integmat()
    center = np.zeros((2,2))
    left = np.zeros((2,2))
    right = np.zeros((2,2))
    for i,j in itertools.product(*[range(2)]*2):
        rcoeff  = 1 if right_edge else 1/2
        lcoeff = 1 if left_edge else 1/2
        vu = vals[i,1]*vals[j,1]*rcoeff - vals[i,0]*vals[j,0]*lcoeff
        dvu =  ints[i,j]
        print(f'center[{i,j}] = vu - dvu = {vu} - {dvu}')
        center[i,j] = vu - dvu
    
        right[i,j] = vals[i,1]*vals[j,0]/2
        left[i,j] = -vals[i,0]*vals[j,1]/2
    return center, right,left
def build_tricol(left_distance:int = 2,right_distance:int = 2):
    c,r,l = tricol(left_edge = left_distance == 0 , right_edge= right_distance == 0)
    print('c = \n',c,'\n r= \n',r,'\n l= \n',l,'\n')
    if left_distance == 1:
        l = l[:-1,]
    if left_distance == 0:
        c = c[:-1,]
    return c,r,l
def get_slices(c,i,j,mult):
    return tuple(slice(mult*ind,mult*ind+c_) for c_,ind in zip(c.shape,(i,j)))
def place_block(mat,block,i,j,mult = 2):
    slc = get_slices(block,i,j,mult = mult)
    mat[slc] = block
    return mat

def build_global(n:int):
    k = 2*n
    x = np.zeros((k,k))
    #left edge
    
    c,r,l = build_tricol(left_distance=0,right_distance=n-1)
    x = place_block(x,c,1,0,mult = 1)
    
    if n==1:
        x[0,:2] = 1,-1
        x[0,2:] = 0
        return x
    
    
    x = place_block(x,r,2,0,mult = 1)
    # x = place_block(x,l,i-1,i)
    
    #interior
    for i in range(1,n-1):
        c,r,l = build_tricol(left_distance=i,right_distance=n-1 -i)
        x = place_block(x,c,i,i)
        x = place_block(x,r,i+1,i)
        x = place_block(x,l,i-1,i)
    
    
        
    c,r,l = build_tricol(left_distance=n-1,right_distance=0)
    x = place_block(x,c,k-c.shape[0],k-c.shape[1],mult = 1)
    # x = place_block(x,r,2,0,mult = 1)
    x = place_block(x,l,k-l.shape[0]-2,k-l.shape[1],mult = 1)
    x = place_block(x,l,k-l.shape[0]-2,k-l.shape[1],mult = 1)
    
    x[0,:2] = 1,-1
    x[0,2:] = 0
    
    return x
def build_rhs(n):
    rhs = np.zeros((2*n))
    rhs[0] = 0
    rhs[1] = 1/n
    for i in range(1,n):
        rhs[2*i] = 1/n
        rhs[2*i+1] = 0
    return rhs
n = 4
x = build_global(n)
rhs = build_rhs(n)
print(x)
print(rhs)

sol = np.linalg.solve(x,rhs)
print(sol)

