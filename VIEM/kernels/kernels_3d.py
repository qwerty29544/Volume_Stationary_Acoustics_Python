import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def kernel_helmholtz_3d(x, y, k=1.0):

    dists = np.sqrt((x - y).dot((x - y)))
    return np.exp(-1j * )