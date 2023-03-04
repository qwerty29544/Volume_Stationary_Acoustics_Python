import numpy as np
import numba as nb
import gmsh
import sys
import json
import os
from VIEM.utils.gmsh_parser import GMSHParser


def compute_volumes_tetrahedron(array_vertexes):
    array_vertexes_first = np.repeat(a=array_vertexes, repeats=[3, 0, 0, 0], axis=1)
    array_vertexes_rest = array_vertexes[:, 1:, :]
    array_vectors_rest_along_first = array_vertexes_rest - array_vertexes_first
    volumes = (1/6) * np.linalg.det(array_vectors_rest_along_first)
    return volumes


def compute_collocations_tetrahedron(array_vertexes):
    return np.mean(array_vertexes, axis=1)


@nb.njit(fastmath=True)
def compute_distances(array_collocations):
    # empty matrix for dists
    dim_len = array_collocations.shape[0]
    matrix_shape = (dim_len, dim_len)
    distances = np.zeros(matrix_shape)
    # applying for all tetrahedrons
    for rut in nb.prange(dim_len - 1):
        for colloc in nb.prange(rut + 1, dim_len):
            dist = array_collocations[rut] - array_collocations[colloc]
            distances[rut][colloc] = np.sqrt(dist.dot(dist))
            distances[colloc][rut] = distances[rut][colloc]
    return distances


# @nb.njit(fastmath=True)
# def n_subdiscr_func(distance, top = 40):
#     return int(np.round(np.exp))


@nb.njit(fastmath=True)
def compute_coeffs_matrix(array_collocations, array_vertexes, kernel_func):
    # empty matrix for filling
    dim_len = array_collocations.shape[0]
    matrix_shape = (dim_len, dim_len)
    matrix_coeffs = np.zeros(matrix_shape) + 1j * np.zeros(matrix_shape)

    # applying func to every element in matrix
    # for rut in nb.prange(dim_len):
    #     for colloc in nb.prange(dim_len):

    return matrix_coeffs


def main():
    print(os.getcwd())
    parser = GMSHParser(file_path=os.path.join("..", "resources", "mesh", "sphere.msh"), dims=3)
    sphere = parser.get_numpy()
    #print(sphere.shape)
    sphere_volumes = compute_volumes_tetrahedron(sphere)
    sphere_collocations = compute_collocations_tetrahedron(sphere)
    sphere_dists = compute_distances(sphere_collocations)
    print(sphere_volumes.shape)
    print(sphere_volumes)
    print(sphere_collocations.shape)
    print("dists", sphere_dists)
    print()
    return 0


if __name__ == "__main__":
    main()
    # array_test = np.array([[[1, 2], [3, 4]], [[1, 3], [5, 6]]])
    # print(array_test)
    # print(np.repeat(array_test, [3, 0], 1))
    # print(np.apply_along_axis(np.linalg.det, 2, array_test))

