# 1d refractions in python
import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def step_refr_1d(collocations_1d, low_bound=-1.0, upper_bound=1.0, refraction=1.0+1.0j):
    """
    Function for generate step refraction function to problem

    :param collocations_1d: **np.ndarray** with **shape=(n,)** - array with generated grid for 1d problem
    :param low_bound: float number with lower bound of 1d problem
    :param upper_bound: float number with upper bound of 1d problem
    :param refraction: complex number (**n + kj**), where **n** is real refraction of complex wave, k - image refraction
    of complex wave
    
    :return: **np.ndarray(dtype=complex)** with **shape=(n,)** - complex array with corresponding refraction to each
    collocation place
    """
    return ((collocations_1d < upper_bound) * (collocations_1d > low_bound)) * refraction + 0.0 + 0.0j


@nb.njit(fastmath=True)
def gauss_refr_1d(collocations_1d, mean_real=0.0, std_real=1.0, mean_imag=1.0, std_imag=1.0):
    """
    Function for generate gaussian refraction function to problem

    :param collocations_1d: **np.ndarray** with **shape=(n,)** - array with generated grid for 1d problem
    :param mean_real: real value for location the top of gaussian of real refraction value
    :param std_real: real value for sqrt of deviation around the top of gaussian of real refraction value
    :param mean_imag: real value for location the top of gaussian of complex refraction value
    :param std_imag: real value for sqrt of deviation around the top of gaussian of imag refraction value

    :return: **np.ndarray(dtype=complex)** with **shape=(n,)** - complex array with corresponding refraction to each
    collocation place
    """
    return (1 / np.sqrt(2 * np.pi * (std_real**2)) * np.exp(-((collocations_1d - mean_real)**2)/(2 * std_real**2)) +
        1j * 1 / np.sqrt(2 * np.pi * (std_imag**2)) * np.exp(-((collocations_1d - mean_imag)**2)/(2 * std_imag**2)))

