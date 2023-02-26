import numpy as np
import numba as nb
import gmsh
import sys
import json
import os
from VIEM.utils.gmsh_parser import GMSHParser


def compute_volumes_tetrahedron(array_vertexes):
    array_vertexes_first = np.repeat(a=array_vertexes, repeats=[3, 0, 0, 0], axis=1)
    array_vertexes_rest = array_vertexes_first[:, 1:, :]
    array_vectors_rest_along_first = array_vertexes_rest - array_vertexes_first
    volumes = (1/6) * np.linalg.det()
    return 0

def main():
    print(os.getcwd())
    parser = GMSHParser(file_path=os.path.join("..", "resources", "mesh", "sphere.msh"), dims=3)
    sphere = parser.get_numpy()
    print(sphere.shape)
    print(np.repeat(sphere, [3, 0, 0, 0], 1))
    return 0


if __name__ == "__main__":
    # main()
    array_test = np.array([[[1, 2], [3, 4]], [[1, 3], [5, 6]]])
    print(array_test)
    print(np.repeat(array_test, [3, 0], 1))
    print(np.apply_along_axis(np.linalg.det, 2, array_test))

