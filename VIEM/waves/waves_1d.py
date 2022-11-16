import numpy as np
import numba as nb


# Функция вектора свободных членов
def wave_narmonic_1d(x, k=1.0, e=1.0, orientation=1.0):
    """
    incoming harmonic wave with monochromatic spectre f(x) = e * (isin(kx) + cos(ikx))

    :param x: **np.ndarray** with **shape=(n,)** of collocation locations
    :param k: real wave number
    :param e: real wave amplitude
    :param orientation: positive or negative orientation of incoming wave
    :return: **np.ndarray** with **shape=(n,)** with values of incoming wave in collocation points
    """
    return e * np.exp(np.sign(orientation) * 1j * k * x)