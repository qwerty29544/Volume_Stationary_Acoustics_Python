import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from VIEM.refractions.refractions_1d import step_refr_1d, gauss_refr_1d
from VIEM.kernels.kernels_1d import kernel_helmholtz_1d_neg
from VIEM.utils.weights import exp_weight, linear_weight
from VIEM.waves.waves_1d import wave_harmonic_1d
from VIEM.fourier_mul.fourier_mul_1d import fourier_complex_matrix_vector_1d
from VIEM.shapes.linear_1d import linear_grid
from VIEM.iterations.two_sdg_symmetric_1d import TwoSGD_nu_1d_sim
from VIEM.utils.config_1d import Problem1dAcoustic


def test_refr():
    N = 1000
    grid_1d = np.linspace(-2.0, 2.0, N)
    gauss_refr = gauss_refr_1d(grid_1d, 1.0, 0.2, -0.5, 0.6)
    step_refr = step_refr_1d(grid_1d, low_bound=-1.5, upper_bound=1.5, refraction=1.0+0.7j) + \
        step_refr_1d(grid_1d, low_bound=-1.0, upper_bound=0.5, refraction=-0.5-0.3j)
    assert gauss_refr.shape[0] == N
    assert step_refr.shape[0] == N
    assert np.all(np.real(gauss_refr) >= 0.0)
    assert np.all(np.imag(gauss_refr) >= 0.0)
    assert np.all(np.real(step_refr) >= 0.0)
    assert np.all(np.imag(step_refr) >= 0.0)


def test_kernel():
    N = 1000
    grid_1d = np.linspace(-2.0, 2.0, N)
    matrix_kernel = kernel_helmholtz_1d_neg(grid_1d[:, None], grid_1d[None, :])
    assert matrix_kernel.shape == (N, N)


def test_weights():
    N = 1000
    grid_1d = np.linspace(0, 10.0, N)
    plt.plot(grid_1d, exp_weight(grid_1d))
    plt.plot(grid_1d, linear_weight(grid_1d, 8., 2., 2.0))
    plt.savefig("weights.png")


def test_fourier_mul():
    N = 10
    k = 1.0
    e = 1.0
    grid_1d = np.linspace(0, 10.0, N)
    matrix_row = kernel_helmholtz_1d_neg(grid_1d[0, None], grid_1d[None, :], k)[0]
    matrix = kernel_helmholtz_1d_neg(grid_1d[:, None], grid_1d[None, :], k)
    vec = wave_harmonic_1d(grid_1d, k, e)
    print(matrix_row)
    res1 = matrix @ vec
    res2 = fourier_complex_matrix_vector_1d(matrix_row, vec)

    print(np.isclose(matrix_row, matrix[0, :]))
    print(res1)
    print(res2)
    print(np.sum(np.abs(res1 - res2)))
    return 0


def test_grid():
    collocations, _, h = linear_grid(-2, 2, 100)
    print(collocations)
    print(h)


def problem_1d():
    problem_test = Problem1dAcoustic(filename="../resources/config_1d.json")
    problem_test.set_problem()
    problem_test.compute_problem()
    problem_test.save_results()
    return 0


if __name__ == "__main__":
    #test_refr()
    #test_kernel()
    #test_weights()
    #test_fourier_mul()
    #test_grid()
    problem_1d()