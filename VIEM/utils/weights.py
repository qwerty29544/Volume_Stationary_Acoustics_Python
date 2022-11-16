import numpy as np
import numba as nb


# Exponential weights for kernel integration in matrix coefficients
@nb.njit(fastmath=True)
def exp_weight(x, upper=10.0, lower=1.0, alpha=0.5):
    return (upper - lower) * np.exp(-alpha * x) + lower


# Linear weights for kernel integration in matrix coefficients
@nb.jit(fastmath=True)
def linear_weight(x, upper=10.0, lower=1.0, alpha=1.):
    return (upper - x * alpha) * ((x * alpha) <= (upper - lower)) + lower * ((x * alpha) > (upper - lower))