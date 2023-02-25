import numpy as np
import numba as nb
import gmsh
import sys
import json
from VIEM.utils.gmsh_parser import GMSHParser


if __name__ == "__main__":
    parser = GMSHParser("/home/leonblue/PycharmProjects/Volume_Stationary_Acoustics_Python/resources/mesh/sphere.msh", 3)
    sphere = parser.get_numpy()
    print(sphere)