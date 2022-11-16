import numpy as np
import numba as nb


# https://ru.wikipedia.org/wiki/%D0%A3%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%93%D0%B5%D0%BB%D1%8C%D0%BC%D0%B3%D0%BE%D0%BB%D1%8C%D1%86%D0%B0
@nb.njit(fastmath=True)
def kernel_helmholtz_1d_pos(x, y, k=1.0):
    """
    function to compute Helmholtz kernel for integral equation
    :param x: **np.ndarray** with **shape=(n,)**
    :param y:
    :param k:
    :return:
    """
    return np.exp(1.0j * k * np.abs(x - y)) / (2.0 * 1.0j * k)


@nb.njit(fastmath=True)
def kernel_helmholtz_1d_neg(x, y, k=1.0):
    """

    :param x:
    :param y:
    :param k:
    :return:
    """
    return -1.0 * np.exp(-1.0j * k * np.abs(x - y)) / (2.0 * 1.0j * k)


