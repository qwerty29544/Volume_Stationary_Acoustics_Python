import numpy as np
import numba as nb
from VIEM.iterations.iter_utils import dot_complex


@nb.njit(fastmath=True)
def IMRES_nu(matrix_A,                # Квадратная матрица оператора
             vector_f,                # Внешний вектор
             vector_nu=None,          # Неоднородность задачи
             vector_u0=None,          # Начальное положение искомого вектора
             eps=10e-07,              # Точность решения задачи
             max_iter=2000):          # Максимальное количество итераций
    # Инициализация начальной переменной
    if vector_u0 is None:
        vector_u0 = np.ones(vector_f.shape[0]) + 0j
    vector_u1 = vector_u0
    # Инициализация неоднородностей
    if vector_nu is None:
        vector_nu = np.ones(vector_f.shape[0]) + 0j
    # Итерационный процесс
    k = 0
    iters = []
    accuracy_iters = []
    resid_iters = []
    for iter in np.arange(max_iter):
        h = vector_u0 + matrix_A @ (vector_nu * vector_u0) - vector_f
        Ah = h + matrix_A @ (vector_nu * h)
        tau = dot_complex(h, Ah) / dot_complex(Ah, Ah)
        vector_u1 = vector_u0 - tau * h
        k += 2
        iters.append(k)
        delta_u = vector_u1 - vector_u0
        accuracy = dot_complex(delta_u, delta_u) / dot_complex(vector_f, vector_f)
        accuracy_iters.append(accuracy)
        resid_iters.append(np.real(np.sqrt(dot_complex(h, h))))
        if (np.real(accuracy) < eps):
            break
        vector_u0 = vector_u1
    return vector_u1, iters, accuracy_iters, resid_iters