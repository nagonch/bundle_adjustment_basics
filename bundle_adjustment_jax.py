import jax.numpy as jnp
from bundle_adjustment import read_bal_data, visualize_data
import os
import urllib
from jax.scipy.spatial.transform import Rotation as R
import jax
import optax
import numpy as np

jax.config.update("jax_enable_x64", True)


def get_x_vector(camera_params, points_3d):
    return jnp.hstack((camera_params.ravel(), points_3d.ravel()))


def get_params_and_points(x_vector, n_cameras, n_points):
    camera_params = x_vector[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = x_vector[n_cameras * 9 :].reshape((n_points, 3))
    return camera_params, points_3d


def project(points, camera_params):
    rotations = R.from_rotvec(camera_params[:, :3]).as_matrix()
    points_proj = jnp.matvec(rotations, points)
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, jnp.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = jnp.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, jnp.newaxis]
    return points_proj


def loss(
    x_vector,
    camera_indices,
    point_indices,
    points_2d,
    n_cameras,
    n_points,
    aggregate_loss=True,
):
    camera_params, points_3d = get_params_and_points(x_vector, n_cameras, n_points)
    projected_points = project(points_3d[point_indices], camera_params[camera_indices])
    error = (projected_points - points_2d) ** 2
    if aggregate_loss:
        error = error.sum()
    return error


def optimize_GD(
    camera_params,
    points_3d,
    camera_indices,
    point_indices,
    points_2d,
    n_cameras,
    n_points,
    learning_rate=1e-2,
    ftol=1e-4,
    max_iter=1000,
):
    x_vector = get_x_vector(camera_params, points_3d)
    forward = jax.value_and_grad(loss)
    loss_prev = loss(
        x_vector, camera_indices, point_indices, points_2d, n_cameras, n_points
    )
    print(f"loss start: {loss_prev:.2e}")
    loss_prev += 2 * ftol * loss_prev
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x_vector)
    for i in range(max_iter):
        loss_val, gradient = forward(
            x_vector, camera_indices, point_indices, points_2d, n_cameras, n_points
        )
        loss_drop = jnp.abs(loss_prev - loss_val)
        updates, opt_state = optimizer.update(gradient, opt_state)
        x_vector = optax.apply_updates(x_vector, updates)
        print(f"{i} loss: {loss_val:.4e}, {ftol * loss_val:.4e}, {loss_drop:.4e}")
        if loss_drop <= ftol * loss_val:
            break
        else:
            loss_prev = loss_val
    camera_params_optimized, points_3d_optimized = get_params_and_points(
        x_vector, n_cameras, n_points
    )
    return camera_params_optimized, points_3d_optimized


if __name__ == "__main__":
    # LOAD DATA
    dataset_url = "https://grail.cs.washington.edu/projects/bal/data/dubrovnik/problem-88-64298-pre.txt.bz2"
    filename = "dubrovnik.txt.bz2"
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(dataset_url, filename)

    data = read_bal_data(filename)
    camera_params, points_3d, camera_indices, point_indices, points_2d = [
        jnp.array(array, dtype=jnp.float64) for array in data
    ]
    camera_indices, point_indices = camera_indices.astype(
        jnp.int32
    ), point_indices.astype(jnp.int32)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    camera_params_optimized, points_3d_optimized = optimize_LM(
        camera_params,
        points_3d,
        camera_indices,
        point_indices,
        points_2d,
        n_cameras,
        n_points,
    )
    np.save("result_points_gd.npy", points_3d_optimized)
    np.save("camera_params_gd.npy", camera_params_optimized)
