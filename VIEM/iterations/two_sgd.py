import numpy as np
import numba as nb
from VIEM.iterations.iter_utils import dot_complex



@nb.njit(fastmath=True)
def TwoSGD(matrix_A,  # Квадратная матрица оператора
           vector_f,  # Внешний вектор
           vector_nu=None,  # Неоднородность задачи
           vector_u0=None,  # Начальное положение искомого вектора
           eps=10e-07,  # Точность решения задачи
           max_iter=2000):  # Максимальное количество итераций)
    # Инициализация начальной переменной
    if vector_u0 is None:
        vector_u0 = np.ones(vector_f.shape[0]) + 0.0j * np.zeros(vector_f.shape[0])
    vector_u1 = np.copy(vector_u0)

    # Инициализация неоднородностей
    if vector_nu is None:
        vector_nu = np.ones(vector_f.shape[0]) + 0.0j * np.ones(vector_f.shape[0])

    # Итерационный процесс
    vector_r0 = vector_u0 + (matrix_A @ (vector_u0 * vector_nu)) - vector_f  # Вектор невязки
    matrix_As = np.conj(matrix_A)  # Сопряженная матрица
    As_r = vector_r0 + (matrix_As @ vector_r0) * np.conj(vector_nu)  # Преобразованная невязка
    A_As_r = As_r + matrix_A @ (As_r * vector_nu)  # Переход невязки
    # Первый итерационный вектор
    vector_u1 = vector_u0 - \
                (dot_complex(As_r, As_r) / dot_complex(A_As_r, A_As_r)) * \
                As_r
    delta_u = vector_u1 - vector_u0
    k = 3
    accuracy = []
    iterations = []
    resid = []
    iterations.append(k)
    norm_f = dot_complex(vector_f, vector_f)
    accuracy.append(np.real(dot_complex(delta_u, delta_u)) / np.real(norm_f))
    resid.append(np.sqrt(np.real(vector_r0.dot(vector_r0.T))))
    if (accuracy[0] < eps):
        return vector_u1, iterations, accuracy, resid
    vector_u2 = vector_u1
    for iter in nb.prange(max_iter):
        vector_r1 = vector_u1 + matrix_A @ (vector_u1 * vector_nu) - vector_f
        delta_r = vector_r1 - vector_r0  # Разница между невязками
        As_r = vector_r1 + (matrix_As @ vector_r1) * np.conj(vector_nu)
        A_As_r = As_r + matrix_A @ (As_r * vector_nu)
        k += 3  # Умножений матрицы на вектор
        iterations.append(k)
        a1 = dot_complex(delta_r, delta_r)
        a2 = dot_complex(As_r, As_r)
        a3 = dot_complex(A_As_r, A_As_r)
        b1 = 0
        denom = a1 * a3 - a2 * a2
        vector_u2 = vector_u1 - \
                    ((-a2 * a2) * (vector_u1 - vector_u0) + (a1 * a2) * As_r) / denom
        delta_u = vector_u2 - vector_u1
        accuracy_iter = np.real(dot_complex(delta_u, delta_u)) / np.real(norm_f)
        accuracy.append(accuracy_iter)
        resid.append(np.real(np.sqrt(vector_r1.dot(vector_r1.T))))
        # print(accuracy)
        if (accuracy_iter < eps):
            break
        vector_r0 = np.copy(vector_r1)
        vector_u0 = np.copy(vector_u1)
        vector_u1 = np.copy(vector_u2)
    return vector_u2, iterations, accuracy, resid