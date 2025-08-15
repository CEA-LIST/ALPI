import numpy as np
from scipy.spatial import Delaunay

def in_hull(p: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    """
    Tests if points in `p` are in the convex hull of `hull_points`.
    `p` is a (N, K) array of N points in K dimensions.
    `hull_points` is a (M, K) array of M points in K dimensions.
    Returns a boolean array of length N.
    """
    try:
        if not isinstance(hull_points, Delaunay):
            hull = Delaunay(hull_points)
        return hull.find_simplex(p) >= 0
    except (ValueError, TypeError):
        # Error can occur if hull points are co-planar/co-linear
        return np.zeros(len(p), dtype=bool)

def extract_pc_in_box3d(pc: np.ndarray, box3d: np.ndarray) -> tuple:
    """
    Extracts points from a point cloud that fall inside a 3D bounding box.
    pc: (N, 3+) point cloud
    box3d: (8, 3) corners of the 3D box
    Returns the points inside the box and their indices.
    """
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds

def roty(t: float) -> np.ndarray:
    """Creates a rotation matrix for a rotation around the y-axis (radians)."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def project_to_image(points_3d: np.ndarray, proj_mat: np.ndarray) -> np.ndarray:
    """
    Applies perspective projection to 3D points.
    points_3d: (3, N)
    proj_mat: (3, 4)
    Returns projected 2D points (2, N).
    """
    num_pts = points_3d.shape[1]
    points_hom = np.vstack((points_3d, np.ones((1, num_pts))))
    points_2d = proj_mat @ points_hom
    points_2d[:2, :] /= points_2d[2, :]
    return points_2d[:2, :]

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arcsin(np.clip(np.cross(v1_u, v2_u), -1.0, 1.0))

def in_camera_coordinate(t: np.ndarray, size: np.ndarray, heading: float) -> np.ndarray:
    """
    Creates the 8 corners of a 3D bounding box in camera coordinates.
    t: (3,) translation vector (center of the box bottom face)
    size: (3,) [l, w, h] dimensions
    heading: scalar, rotation angle around Y-axis
    Returns (3, 8) array of box corners.
    """
    l, w, h = size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    box_corners = np.vstack([x_corners, y_corners, z_corners])
    
    R = roty(heading)
    points_3d = R @ box_corners
    points_3d += t.reshape(3, 1)
    return points_3d

def estimate_depth_from_box(calib_P: np.ndarray, bbox: np.ndarray, object_height: float) -> float:
    """
    Estimates the depth (Z-coordinate) of an object given its 2D bounding box
    and known real-world height.
    """
    box_height_px = bbox[3] - bbox[1]
    if box_height_px <= 0: return 0.0
    focal_y = calib_P[1, 1]
    return (focal_y * object_height) / box_height_px

def extract_pc_in_frustum(pc_rect, pc_image_coord, img_fov_inds, box2d, calib):
    """
    Extracts points within a 2D box's frustum and calculates the frustum angle.
    """
    xmin, ymin, xmax, ymax = box2d
    box_fov_inds = (pc_image_coord[:, 0] >= xmin) & \
                   (pc_image_coord[:, 0] < xmax) & \
                   (pc_image_coord[:, 1] >= ymin) & \
                   (pc_image_coord[:, 1] < ymax)
    box_fov_inds = box_fov_inds & img_fov_inds
    pc_in_box_fov = pc_rect[box_fov_inds, :]

    # Calculate frustum angle
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # Assume a depth of 20m for angle calculation
    box2d_center_rect = calib.project_image_to_rect(uvdepth)
    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2], box2d_center_rect[0, 0])
    
    return pc_in_box_fov, frustum_angle
