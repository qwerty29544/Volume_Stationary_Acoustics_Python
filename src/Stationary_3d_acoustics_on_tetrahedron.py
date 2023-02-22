import numpy as np
import numba as nb
import gmsh
import sys
import json
from VIEM.utils.gmsh_parser import GMSHParser


class MyConfig:
    def __init__(self, file_path=None):
        self.filepath = "resources/mesh/sphere.msh"
        self.dims = 3
        if file_path is not None:
            self.filepath, self.dims = self._startup_parse(file_path)

    def _startup_parse(self, file_path):
        with open(file_path, "r") as jsonfile:
            data_config = json.load(jsonfile)
        filepath = data_config.get("filepath")
        dims = data_config.get("dims")
        return filepath, dims


def main(config_path=None):
    if config_path is not None:
        config = MyConfig(config_path)
    else:
        config = MyConfig(sys.argv[2])
    parser = GMSHParser(config.filepath, config.dims)
    array_3d = parser.get_numpy()
    print(array_3d[1:10])


if __name__ == "__main__":
    gmsh.open("resources/mesh/sphere.msh")
    #main("/home/leonblue/PycharmProjects/Volume_Stationary_Acoustics_Python/resources/config_3d_tetra.json")