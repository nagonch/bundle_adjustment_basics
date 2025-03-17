import urllib
import os
from bundle_adjustment import read_bal_data
import numpy as np
from scipy.sparse import lil_matrix


def get_jacobian(
    n_cameras,
    n_points,
    camera_indices,
    point_indices,
    dr=None,
    dcamera_params=None,
    dpoint_indices=None,
):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    J = lil_matrix((m, n), dtype=int)

    if dr is None or dcamera_params is None or dpoint_indices is None:
        i = np.arange(camera_indices.size)
        for s in range(9):
            J[2 * i, camera_indices * 9 + s] = 1
            J[2 * i + 1, camera_indices * 9 + s] = 1

        for s in range(3):
            J[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            J[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
    return J


if __name__ == "__main__":
    # LOAD DATA
    dataset_url = "https://grail.cs.washington.edu/projects/bal/data/dubrovnik/problem-88-64298-pre.txt.bz2"
    filename = "dubrovnik.txt.bz2"
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(dataset_url, filename)

    data = read_bal_data(filename)
    camera_params, points_3d, camera_indices, point_indices, points_2d = [
        np.array(array, dtype=np.float64) for array in data
    ]
    camera_indices, point_indices = camera_indices.astype(
        np.int32
    ), point_indices.astype(np.int32)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    get_jacobian(
        n_cameras,
        n_points,
        camera_indices,
        point_indices,
    )
