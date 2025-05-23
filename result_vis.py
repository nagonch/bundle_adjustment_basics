from bundle_adjustment import visualize_data
import numpy as np

points_3d, camera_params = np.load("result_points_gd.npy"), np.load(
    "camera_params_gd.npy"
)
visualize_data(points_3d, camera_params)
