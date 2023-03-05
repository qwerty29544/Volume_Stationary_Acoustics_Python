import numpy as np
import numba as nb


# Функция вектора свободных членов
@nb.njit(fastmath=True)
def wave_harmonic_3d(x, k=1.0, e=1.0, phi0=0.0, orientation=np.array([1.0, 0.0, 0.0])):
    orientation = orientation / np.sqrt(orientation.dot(orientation))
    return e * np.exp(1j * k * x.dot(orientation) + phi0)