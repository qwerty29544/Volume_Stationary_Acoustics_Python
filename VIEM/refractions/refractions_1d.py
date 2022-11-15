# 1d refractions in python
import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def step_refr_1d(collocations_1d, low_bound=-1.0, upper_bound=1.0, refraction_coeff=1.0 + 1.0j):
    return (low_bound < collocations_1d < upper_bound) * refraction_coeff + 0.0 + 0.0j

