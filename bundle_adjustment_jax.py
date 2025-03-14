import jax.numpy as jnp
from bundle_adjustment import read_bal_data, visualize_data
import os
import urllib
from jax.scipy.spatial.transform import Rotation as R
import jax


def get_x_vector(camera_params, points_3d):
    return jnp.hstack((camera_params.ravel(), points_3d.ravel()))


def get_params_and_points(x_vector, n_cameras, n_points):
    camera_params = x_vector[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = x_vector[n_cameras * 9 :].reshape((n_points, 3))
    return camera_params, points_3d


def project(points, camera_params):
    rotations = R.from_euler("xyz", camera_params[:, :3]).as_matrix()
    print(points.shape, rotations.shape)
    points_proj = jnp.matvec(rotations, points)
    print(points_proj)
    raise
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, jnp.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = jnp.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, jnp.newaxis]
    return points_proj


def loss(x_vector, camera_indices, point_indices, points_2d, n_cameras, n_points):
    camera_params, points_3d = get_params_and_points(x_vector, n_cameras, n_points)
    projected_points = project(points_3d[point_indices], camera_params[camera_indices])
    error = (jnp.linalg.norm(projected_points - points_2d, axis=1) ** 2).sum()
    return error


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
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    x_vector = get_x_vector(camera_params, points_3d)
    error = loss(
        x_vector, camera_indices, point_indices, points_2d, n_cameras, n_points
    )
    print(error)
