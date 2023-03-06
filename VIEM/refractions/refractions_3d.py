import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def step_refr_3d_sphere(collocations_3d, center=np.array([0.0, 0.0, 0.0]), radius=1.0, refraction=1.0 + 1.0j):
    return (np.sqrt(np.sum((collocations_3d - center)**2, axis=1)) <= radius) * refraction


@nb.njit(fastmath=True)
def step_refr_3d_cube(collocations_3d, center=np.array([0.0, 0.0, 0.0]), radius=1.0, refraction=1.0 + 1.0j):
    return (np.sum(np.abs(collocations_3d - center), axis=1) <= radius) * refraction


if __name__ == "__main__":
    test_collocs = np.array([[0, 1.0, 1.7], [3.0, -1.0, 0], [0.5, 0.4, 0.3]])
    print(step_refr_3d_cube(test_collocs))
    print(step_refr_3d_sphere(test_collocs))