import jax.numpy as jnp
import urllib
import os
from bundle_adjustment import read_bal_data
import jax

jax.config.update("jax_enable_x64", True)

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
