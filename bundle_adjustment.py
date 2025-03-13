from __future__ import print_function
import urllib.request
import bz2
import os
import numpy as np
import open3d as o3d
import viser
import time
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares


def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))
    return camera_params, points_3d, camera_indices, point_indices, points_2d


def visualize_data(points_3d, camera_params):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)
    _, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)
    points_3d = np.array(point_cloud.points)
    server = viser.ViserServer()
    server.scene.world_axes.visible = True

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    colors = np.zeros_like(points_3d)
    colors[:, 0] = 255
    server.scene.add_point_cloud("my_point_cloud", points_3d, colors, point_size=0.01)

    for i, param in enumerate(camera_params):
        server.scene.add_camera_frustum(
            name=f"{i}",
            aspect=1,
            fov=np.pi / 3,
            scale=0.05,
            wxyz=R.from_euler("xyz", param[:3]).as_quat(),
            position=param[3:6],
        )

    while True:
        time.sleep(2.0)


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return (
        cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    )


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9 :].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


if __name__ == "__main__":
    # LOAD DATA
    dataset_url = "http://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2"
    filename = "dataset.txt.bz2"
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(dataset_url, filename)

    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(
        filename
    )
    # INSPECT DATA
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    # VISUALIZE DATA
    # visualize_data(points_3d, camera_params)

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(
        fun,
        x0,
        jac_sparsity=A,
        verbose=2,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d),
    )
    x = res.x
    t1 = time.time()

    camera_params = x[: 9 * n_cameras].reshape(n_cameras, 9)
    points_3d = x[9 * n_cameras :].reshape(n_points, 3)
    np.save("result_points.npy", points_3d)
    np.save("camera_params.npy", camera_params)
    visualize_data(points_3d, camera_params)
