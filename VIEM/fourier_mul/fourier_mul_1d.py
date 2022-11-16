import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def prep_fourier_my_matrix(A_first):
    """
    Preparation func for fourier mult of simmetric toeplitz matrix and vector. Function transforms first row of toeplitz
    matrix to circulant row.

    :param A_first: **np.ndarray** with **shape=(N,)** - first row of toeplitz simmetric matrix
    :return: A_vec - **np.ndarray** with **shape=(2N,)** - transformed like A_vec = [a_{0}, a_{-1}, .., a_{-N + 1}, 0,
    a_{N - 1}, a_{N-2}, .., a_{1}], len(A) = 2N
    """
    A_vec = np.append(A_first, np.append(np.array([0]), A_first[A_first.shape[0]:0:-1]))
    return A_vec


# #@nb.njit(fastmath=True, parallel=True, cache=True)
@nb.njit(fastmath=True)
def prep_fourier_my_vector(u_vec):
    """
    Preparation func for fourier mult of simmetric toeplitz matrix and vector. Function transforms vector to new vector
    with trailing N zeros on tail

    :param u_vec: input **np.ndarray** vector with **shape=(N, )**
    :return: u_prep - transformed **np.ndarray** vector with **shape=(2N, )** by rule of trailing zeros on tail
    """
    u_prep = np.append(u_vec, np.zeros(u_vec.shape[0]))
    return u_prep


# #@nb.njit(fastmath=True, parallel=True, cache=True)
@nb.njit(fastmath=True, cache=True)
def fourier_mult_1d(A_vec, f_vec, N):
    # Циркулянтная матрица A:
    #      |a_{0}   a_{-1}   ..  a_{-N}   |
    #      |a_{1}   a_{0}    ..  a_{-N+1} |
    #  A = |............................. |
    #      |a_{N-1} a_{N-2}  ..  a_{-1}   |
    #      |a_{N}   a_{N-1}  ..  a_{0}    |

    # A_vec: первая строчка такой матрицы заполненная следующим образом:
    # A_vec = [a_{0}, a_{-1}, .., a_{-N}, 0, a_{N}, a_{N-1}, .., a_{1}], len(A) = 2N + 2
    # f_vec = [f_{0}, f_{1}, .., f{N}, 0, 0, .., 0], len(f) = 2N + 2

    with nb.objmode(A_fft='complex128[:]', f_fft='complex128[:]'):
        A_fft = np.fft.fft(A_vec)
        f_fft = np.fft.fft(f_vec)

    res_fft = A_fft * f_fft

    with nb.objmode(res='complex128[:]'):
        res = np.fft.ifft(res_fft)

    return res[:N]


@nb.njit(fastmath=True)
def fourier_matrix_vector_1d_mul(A_first_row, u_vec):
    return fourier_mult_1d(prep_fourier_my_matrix(A_first_row),
                           prep_fourier_my_vector(u_vec),
                           A_first_row.shape[0])


@nb.njit(fastmath=True, parallel=True)
def fourier_complex_matrix_vector_1d(A_first_row, u_vec):
    N = A_first_row.shape[0]
    A_first_row = prep_fourier_my_matrix(A_first_row)
    u_vec = prep_fourier_my_vector(u_vec)

    return (fourier_mult_1d(np.real(A_first_row), np.real(u_vec), N) +
            fourier_mult_1d(-1.0 * np.imag(A_first_row), np.imag(u_vec), N)) + \
            1.0j * (fourier_mult_1d(np.imag(A_first_row), np.real(u_vec), N) +
                    fourier_mult_1d(np.real(A_first_row), np.imag(u_vec), N))
