import jax.numpy as jnp
from bundle_adjustment import read_bal_data, visualize_data
import os
import urllib
from jax.scipy.spatial.transform import Rotation as R


def project(points, camera_params):
    rotations = R.from_euler("xyz", camera_params[:, :3]).as_matrix()
    points_proj = jnp.matmul(rotations, points[..., None]).squeeze(-1)
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, jnp.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = jnp.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, jnp.newaxis]
    return points_proj


if __name__ == "__main__":
    # LOAD DATA
    dataset_url = "https://grail.cs.washington.edu/projects/bal/data/dubrovnik/problem-88-64298-pre.txt.bz2"
    filename = "dubrovnik.txt.bz2"
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(dataset_url, filename)

    data = read_bal_data(filename)
    camera_params, points_3d, camera_indices, point_indices, points_2d = [
        jnp.array(array) for array in data
    ]
    projected_points = project(points_3d[point_indices], camera_params[camera_indices])
