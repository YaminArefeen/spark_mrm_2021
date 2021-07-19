import numpy as np
import cupy as cp
from scipy.sparse import linalg as sla 

def conjgradscipy(normal,x0,yadj,iterations): #Numpy implementation of conjugate gradients
    P,N = yadj.shape

    def mv(x):
        return normal(x.reshape((P,N))).ravel()

    AhA = sla.LinearOperator((P*N,P*N),matvec = mv,rmatvec = mv)
    x,_ = sla.cg(AhA,yadj.ravel(),x0=x0.ravel(),maxiter=iterations)
    return x.reshape((P,N))

def conjgrad(normal,x,yadj,ite): #My implementation of conjugate gradients, allowing cupy implementation
    '''
    Implementation of basic conjugate gradients.  Will allow CUDA compatibility.
    Inputs:
        normal - Function handle which will compute the normal operator A^H A 
        x      - N x 1, Initial guess at a solution
        yadj   - N x 1, Acquired data that we want to match
        ite    - Number of iterations for the conjugate gradient method
    Outputs: 
        res    - N x 1, recovered vector from the conjugate gradient algorithm
    '''
    xp = cp.get_array_module(x)

    r = yadj - normal(x)
    p = xp.copy(r)

    ro = xp.inner(xp.conj(r),r)

    for k in range(ite):
        Ap = normal(p)

        ak = ro/xp.inner(xp.conj(p),Ap)
        x  = x + ak * p
        r  = r - ak * Ap
        
        rn = xp.inner(xp.conj(r),r)
        p  = r + rn/ro * p

        ro = xp.copy(rn)

    return x

