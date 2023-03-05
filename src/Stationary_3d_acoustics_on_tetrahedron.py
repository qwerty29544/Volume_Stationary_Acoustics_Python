import numpy as np
import numba as nb
import gmsh
import sys
import json
import matplotlib.pyplot as plt
import os
from VIEM.utils.gmsh_parser import GMSHParser
from VIEM.kernels.kernels_3d import kernel_helmholtz_3d
import VIEM.refractions.refractions_3d as refrs_3d
from VIEM.waves.waves_3d import wave_harmonic_3d
from VIEM.iterations.two_sgd import TwoSGD


plt.rcParams.update({'font.size': 18})


def compute_volumes_tetrahedron(array_vertexes_3d):
    array_vertexes_first = np.repeat(a=array_vertexes_3d, repeats=[3, 0, 0, 0], axis=1)
    array_vertexes_rest = array_vertexes_3d[:, 1:, :]
    array_vectors_rest_along_first = array_vertexes_rest - array_vertexes_first
    volumes = (1/6) * np.linalg.det(array_vectors_rest_along_first)
    return volumes


def compute_collocations_tetrahedron(array_vertexes):
    return np.mean(array_vertexes, axis=1)


@nb.njit(fastmath=True)
def compute_distances(collocations):
    # empty matrix for dists
    dim_len = collocations.shape[0]
    matrix_shape = (dim_len, dim_len)
    distances = np.zeros(matrix_shape)
    # applying for all tetrahedrons
    for rut in nb.prange(dim_len - 1):
        for colloc in nb.prange(rut + 1, dim_len):
            dist = collocations[rut] - collocations[colloc]
            distances[rut][colloc] = np.sqrt(dist.dot(dist))
            distances[colloc][rut] = distances[rut][colloc]
    return distances


# @nb.njit(fastmath=True)
# def n_subdiscr_func(distance, top = 40):
#     return int(np.round(np.exp))


@nb.njit(fastmath=True)
def compute_coeffs_matrix_simple(collocations_3d, kernel_func, k=1.0):
    # empty matrix for filling
    dim_len = collocations_3d.shape[0]
    matrix_shape = (dim_len, dim_len)
    matrix_coeffs = np.zeros(matrix_shape) + 1j * np.zeros(matrix_shape)

    # applying func to every element in matrix
    for rut in nb.prange(dim_len - 1):
        for colloc in nb.prange(rut + 1, dim_len):
            matrix_coeffs[rut, colloc] = kernel_func(collocations_3d[rut], collocations_3d[colloc], k)
            matrix_coeffs[colloc, rut] = matrix_coeffs[rut, colloc]

    return matrix_coeffs


def main():
    config = {
        "k": 2.0,
        "d": [1.0, 0.5, 0.5],
        "E0": 1.0,
        "path_to_mesh": "..\\resources\\mesh\\sphere.msh"
    }

    k = config.get("k", 1.0)
    vector = np.array(config.get("d", [1.0, 0.0, 0.0]))
    vector = vector / np.sqrt(vector.dot(vector))


    parser = GMSHParser(file_path=os.path.join("..", "resources", "mesh", "sphere.msh"), dims=3)
    sphere = parser.get_numpy()
    sphere_volumes = compute_volumes_tetrahedron(sphere)
    sphere_collocations = compute_collocations_tetrahedron(sphere)
    sphere_coeffs = compute_coeffs_matrix_simple(sphere_collocations, kernel_helmholtz_3d, k)
    sphere_dists = compute_distances(sphere_collocations)
    sphere_refractions = (
        refrs_3d.step_refr_3d_cube(
            collocations_3d=sphere_collocations,
            center=np.array([0.2, 0.2, 0.2]),
            radius=0.4,
            refraction=2.0 + 2.0j
        ) +
        refrs_3d.step_refr_3d_sphere(
            collocations_3d=sphere_collocations,
            center=np.array([0.0, 0.0, 0.0]),
            radius=10.0,
            refraction=1.0 + 0.0j
        ) -
        1.0
    )
    free_vec = wave_harmonic_3d(sphere_collocations, k=4.0)
    G_matrix = sphere_coeffs * sphere_volumes

    result_TwoSGD, iters_TwoSGD, accuracy_TwoSGD, resid_TwoSGD = TwoSGD(matrix_A=(-k**2) * G_matrix,
                                                                        vector_f=(-1 * G_matrix) @ free_vec)

    # График итераций на норме ве
    plt.figure(figsize=(12, 10), dpi=100)
    plt.plot(iters_TwoSGD, accuracy_TwoSGD, color='#AA2200', label="TwoSGD")
    plt.xlabel("Количество умножений матрицы на вектор")
    plt.ylabel("Норма относительного изменения приближения")
    plt.title("Сходимость итерационых методов")
    plt.legend()
    plt.savefig("..\\resources\\figures\\iterations_accuracy.png")

    plt.figure(figsize=(12, 10), dpi=100)
    plt.plot(iters_TwoSGD, resid_TwoSGD, color='#AA2200', label="TwoSGD")
    plt.xlabel("Количество умножений матрицы на вектор")
    plt.ylabel("Норма невязки на итерации")
    plt.title("Сходимость итерационых методов")
    plt.legend()
    plt.savefig("..\\resources\\figures\\iterations_resid.png")

    return 0


if __name__ == "__main__":
    main()
    # array_test = np.array([[[1, 2], [3, 4]], [[1, 3], [5, 6]]])
    # print(array_test)
    # print(np.repeat(array_test, [3, 0], 1))
    # print(np.apply_along_axis(np.linalg.det, 2, array_test))

