import numpy as np
from numba import njit

@njit
def f(x, y, pars):
    # polynomial map function
    fx = pars[0] + pars[1]*x + pars[2]*y + pars[3]*x*x + pars[4]*x*y + pars[5]*y*y
    fy = pars[6] + pars[7]*x + pars[8]*y + pars[9]*x*x + pars[10]*x*y + pars[11]*y*y
    return fx, fy

@njit
def generate_iterates(max_its, pars):
    x = 0.0
    y = 0.0
    Fx = np.empty(max_its, dtype=np.float64)
    Fy = np.empty(max_its, dtype=np.float64)

    for i in range(max_its):
        x, y = f(x, y, pars)
        Fx[i] = x
        Fy[i] = y
        r = x*x + y*y
        if r > 4.0:
            # truncate output if escape condition reached
            return Fx[:i], Fy[:i]
    return Fx, Fy
