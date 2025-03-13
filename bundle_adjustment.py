from __future__ import print_function
import urllib.request
import bz2
import os
import numpy as np
import open3d as o3d
import viser
import time
from scipy.spatial.transform import Rotation as R


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
    server.scene.add_point_cloud("my_point_cloud", points_3d, colors, point_size=0.05)

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
    visualize_data(points_3d, camera_params)
