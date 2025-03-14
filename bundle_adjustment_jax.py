import jax.numpy as jnp
from bundle_adjustment import read_bal_data, visualize_data
import os
import urllib

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
    print(
        camera_params.shape,
        points_3d.shape,
        camera_indices.shape,
        point_indices.shape,
        points_2d.shape,
    )
