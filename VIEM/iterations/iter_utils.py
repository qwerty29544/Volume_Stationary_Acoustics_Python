import numpy as np
import numba as nb


# Скалярное произведение комл. для бисопряженных градиентов
@nb.njit(fastmath=True)
def B_compl_dot(vec1, vec2):
    return np.conj(vec1).dot(vec2)


# Скалярное произведение реал. для бисопряженных градиентов
@nb.njit(fastmath=True)
def B_stand_dot(vec1, vec2):
    return vec1.dot(vec2)


# Функция скалярного произведения комплексных векторов
@nb.njit(fastmath=True)
def dot_complex(vector_c1, vector_c2):
    return vector_c1.dot(np.conj(vector_c2))