from bundle_adjustment import *

if __name__ == "__main__":
    # LOAD DATA
    dataset_url = "https://grail.cs.washington.edu/projects/bal/data/dubrovnik/problem-88-64298-pre.txt.bz2"
    filename = "dubrovnik.txt.bz2"
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
    try:
        visualize_data(points_3d, camera_params)
    except KeyboardInterrupt:
        pass
