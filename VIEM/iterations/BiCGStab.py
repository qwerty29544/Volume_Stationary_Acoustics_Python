import numpy as np
import numba as nb
from VIEM.iterations.iter_utils import B_compl_dot, B_stand_dot, dot_complex


# Стабилизированный метод бисопряженных градиентов
def BiCGStab_nu(matrix_A,                # Квадратная матрица оператора
                vector_f,                # Внешний вектор
                vector_nu=None,          # Неоднородность задачи (рефракция)
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
    r_0 = vector_f - (vector_u0 + matrix_A @ (vector_nu * vector_u0))
    r_tild = r_0
    rho_0 = 1
    alpha_0 = 1
    omega_0 = 1
    v_0 = np.zeros(vector_f.shape[0])
    p_0 = np.zeros(vector_f.shape[0])
    k = 1
    iters = []
    accuracy_iters = []
    resid_iters = []
    for iter in np.arange(max_iter):
        rho_1 = B_compl_dot(r_tild, r_0)
        beta_1 = (rho_1 / rho_0) * (alpha_0 / omega_0)
        p_1 = r_0 + beta_1 * (p_0 - omega_0 * v_0)
        v_1 = p_1 + matrix_A @ (vector_nu * p_1) # A @ p_1 == p_1 + A @ (p_1 * nu)
        alpha_1 = rho_1 / B_compl_dot(r_tild, v_1)
        s_1 = r_0 - alpha_1 * v_1
        t_1 = s_1 + matrix_A @ (vector_nu * s_1) # A @ s_1 == s_1 + A @ (s_1 * nu)
        omega_1 = B_stand_dot(t_1, s_1) / B_stand_dot(t_1, t_1)
        vector_u1 = vector_u0 + omega_1 * s_1 + alpha_1 * p_1
        r_1 = s_1 - omega_1 * t_1
        k += 2
        iters.append(k)
        delta_u = vector_u1 - vector_u0
        accuracy = dot_complex(delta_u, delta_u) / dot_complex(vector_f, vector_f)
        accuracy_iters.append(np.real(accuracy))
        resid = vector_u1 + (matrix_A @ (vector_u1 * vector_nu)) - vector_f
        resid_iters.append(np.real(np.sqrt(dot_complex(resid, resid))))
        if (np.real(accuracy) < eps):
            break
        vector_u0 = vector_u1
        rho_0 = rho_1
        alpha_0 = alpha_1
        omega_0 = omega_1
        v_0 = v_1
        p_0 = p_1
        r_0 = r_1
    return vector_u1, iters, accuracy_iters, resid_iters
