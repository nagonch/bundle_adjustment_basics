import urllib
import os
from bundle_adjustment import read_bal_data
import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R
from scipy.sparse.linalg import lsqr
import scipy.sparse as sp
import matplotlib.pyplot as plt


def get_x_vector(camera_params, points_3d):
    return np.hstack((camera_params.ravel(), points_3d.ravel()))


def get_params_and_points(x_vector, n_cameras, n_points):
    camera_params = x_vector[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = x_vector[n_cameras * 9 :].reshape((n_points, 3))
    return camera_params, points_3d


def project(points, camera_params):
    rotations = R.from_rotvec(camera_params[:, :3]).as_matrix()
    points_proj = np.matvec(rotations, points)
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
    params, points = get_params_and_points(params, n_cameras, n_points)
    points_proj = project(points[point_indices], params[camera_indices])
    result = (points_proj - points_2d).ravel()
    return result


def get_jacobian(
    n_cameras,
    n_points,
    camera_indices,
    point_indices,
    dr=None,
    dx=None,
    eps=1e-9,
):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    J = lil_matrix((m, n), dtype=float)

    if dr is None or dx is None:
        dr = np.ones(shape=(points_2d.shape[0] * 2,))
        dx = np.ones(shape=(n_cameras * 9 + n_points * 3,))

    for i in range(m):
        start_column = camera_indices[i // 2] * 9
        J[i, start_column : start_column + 9] = dr[i] / (
            dx[start_column : start_column + 9] + eps
        )

        start_column = point_indices[i // 2] * 3
        J[i, n_cameras * 9 + start_column : n_cameras * 9 + start_column + 3] = dr[
            i
        ] / (dx[n_cameras * 9 + start_column : n_cameras * 9 + start_column + 3] + eps)
    return J


def skew(v):
    """Skew-symmetric matrix for 3D vector"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(omega):
    """Rotation matrix from Rodrigues vector"""
    theta = np.linalg.norm(omega)
    if theta < 1e-6:
        return np.eye(3)
    k = omega / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def get_jacobian(
    n_cameras, n_points, camera_indices, point_indices, cameras, points, eps=1e-9
):
    """
    Parameters:
    cameras: array of shape (n_cameras, 9) containing:
        [omega (3), t (3), f (1), cx (1), cy (1)]
    points: array of shape (n_points, 3) - 3D coordinates
    """
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    J = lil_matrix((m, n), dtype=np.float64)

    for i in range(m):
        k = i // 2
        cam_idx = camera_indices[k]
        pt_idx = point_indices[k]

        omega = cameras[cam_idx, :3]
        t = cameras[cam_idx, 3:6]
        f = cameras[cam_idx, 6]
        cx = cameras[cam_idx, 7]
        cy = cameras[cam_idx, 8]
        point = points[pt_idx]

        R = rodrigues(omega)
        X_cam = R @ point + t
        X_, Y_, Z_ = X_cam
        inv_Z = 1.0 / (Z_ + eps)
        x = X_ * inv_Z
        y = Y_ * inv_Z

        # Common terms
        f_inv_Z = f * inv_Z
        f_X_inv_Z2 = f * X_ * inv_Z**2
        f_Y_inv_Z2 = f * Y_ * inv_Z**2

        if i % 2 == 0:  # u-component
            du_dXcam = np.array([f_inv_Z, 0, -f_X_inv_Z2])
        else:  # v-component
            dv_dXcam = np.array([0, f_inv_Z, -f_Y_inv_Z2])

        # Rotation derivatives
        p_rotated = R @ point
        skew_p = skew(p_rotated)
        dXcam_domega = -R @ skew_p

        # Translation derivatives (identity matrix)
        dXcam_dt = np.eye(3)

        # Point derivatives
        dXcam_dpoint = R

        if i % 2 == 0:
            # Camera parameter derivatives
            du_domega = du_dXcam @ dXcam_domega
            du_dt = du_dXcam @ dXcam_dt
            du_df = x
            du_dcx = 1.0
            du_dcy = 0.0

            # Point derivatives
            du_dpoint = du_dXcam @ dXcam_dpoint

            J[i, cam_idx * 9 : cam_idx * 9 + 9] = [
                *du_domega,
                *du_dt,
                du_df,
                du_dcx,
                du_dcy,
            ]
            J[i, n_cameras * 9 + pt_idx * 3 : n_cameras * 9 + pt_idx * 3 + 3] = (
                du_dpoint
            )
        else:
            # Camera parameter derivatives
            dv_domega = dv_dXcam @ dXcam_domega
            dv_dt = dv_dXcam @ dXcam_dt
            dv_df = y
            dv_dcx = 0.0
            dv_dcy = 1.0

            # Point derivatives
            dv_dpoint = dv_dXcam @ dXcam_dpoint

            J[i, cam_idx * 9 : cam_idx * 9 + 9] = [
                *dv_domega,
                *dv_dt,
                dv_df,
                dv_dcx,
                dv_dcy,
            ]
            J[i, n_cameras * 9 + pt_idx * 3 : n_cameras * 9 + pt_idx * 3 + 3] = (
                dv_dpoint
            )

    return J


def get_opt_x_LM(
    camera_params,
    points_3d,
    camera_indices,
    point_indices,
    points_2d,
    n_cameras,
    n_points,
    ftol=1e-4,
    max_iter=1000,
    mu=1e-3,
):
    x_params = get_x_vector(camera_params, points_3d)
    residual = res_prev = fun(
        x_params, n_cameras, n_points, camera_indices, point_indices, points_2d
    )
    J = 1 * get_jacobian(
        n_cameras,
        n_points,
        camera_indices,
        point_indices,
        camera_params,
        points_3d,
    )
    loss_prev = (res_prev**2).sum()
    print(f"loss start: {loss_prev:.5e}")
    loss_prev += 2 * ftol * loss_prev
    for i in range(max_iter):
        JTJ = J.T @ J
        JTr = J.T @ residual
        delta = lsqr(JTJ + mu * sp.eye(JTJ.shape[0]), -JTr)[0]
        x_new = x_params - delta
        residual = fun(
            x_new, n_cameras, n_points, camera_indices, point_indices, points_2d
        )
        res_prev = residual
        loss_val = (residual**2).sum()
        print(f"{i}, {loss_val:.5e}, {mu:.3e}")
        loss_drop = loss_prev - loss_val
        if loss_drop <= 0:
            mu *= 10
        else:
            mu /= 10
            x_params = x_new
        loss_prev = loss_val
        camera_params, points_3d = get_params_and_points(x_new, n_cameras, n_points)
        J = get_jacobian(
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
            camera_params,
            points_3d,
        )

    return x_params


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
    # N_POINTS = points_2d.shape[0]
    N_POINTS = 10000
    inds = np.arange(points_2d.shape[0])
    np.random.shuffle(inds)
    inds = np.load("inds.npy")
    inds = inds[:N_POINTS]
    points_2d = points_2d[inds]
    camera_indices = camera_indices[inds]
    point_indices = point_indices[inds]
    camera_indices, point_indices = camera_indices.astype(
        np.int32
    ), point_indices.astype(np.int32)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    dr = np.ones(shape=(points_2d.shape[0] * 2,))
    dcamera_params = np.ones(shape=(n_cameras * 9,))
    dpoint_values = np.ones(shape=(n_points * 3,))
    x_vector = get_x_vector(camera_params, points_3d)
    x_opt = get_opt_x_LM(
        camera_params,
        points_3d,
        camera_indices,
        point_indices,
        points_2d,
        n_cameras,
        n_points,
    )
    # print(get_params_and_points(x_vector, n_cameras, n_points))
