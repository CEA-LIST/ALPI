import numpy as np
from typing import Dict, Any


import geometry
from draw_util import get_lidar_in_image_fov
from priors import ObjectPriors


APP_POS = {
    'Car': np.array([-0.1, 0.6, 0.8]),
    'Pedestrian': np.array([0, 0.8, 0]),
    'Cyclist': np.array([-0.29, -0.7, -0.3]),
}
FAKE_OBJECT_RANGE = {
    'Car': (4.0, 2.0),
    'Pedestrian': (1.5, 1.5),
    'Cyclist': (2.0, 2.0),
}

# --- Data Structures ---
class SceneData:
    """A container for per-frame KITTI data to simplify passing arguments."""
    def __init__(self, dataset, data_idx: int):
        """
        Loads and pre-processes all necessary data for a single scene/frame.
        
        Args:
            dataset: An instance of the kitti_object dataset class.
            data_idx: The index of the data frame to load.
        """
        self.calib = dataset.get_calibration(data_idx)
        self.img = dataset.get_image(data_idx)
        self.img_shape = self.img.shape
        pc_velo = dataset.get_lidar(data_idx)
        
        # Project LiDAR points to camera coordinates
        self.pc_rect = self.calib.project_velo_to_rect(pc_velo[:, 0:3])
        self.pc_rect = np.hstack((self.pc_rect, pc_velo[:, 3:]))
        
        # Get LiDAR points that are within the field of view of the image
        _, self.pc_image_coord, self.img_fov_inds = get_lidar_in_image_fov(
            pc_velo[:, 0:3], self.calib, 0, 0, self.img_shape[1], self.img_shape[0], True
        )



def initialize_data_dict(keys: list) -> Dict[str, list]:
    """
    Initializes a dictionary with empty lists for the given keys.
    
    Args:
        keys: A list of strings representing the data categories.
        
    Returns:
        A dictionary with keys formatted as 'key_list' and empty list values.
    """
    return {f"{key}_list": [] for key in keys}

def append_data(data_dict: Dict[str, list], **kwargs: Any) -> None:
    """
    Appends a set of values to the corresponding lists in a data dictionary.
    This function simplifies the process of populating the output data structure.
    
    Args:
        data_dict: The dictionary containing data lists.
        **kwargs: Keyword arguments where the key is the data category (e.g., 'id')
                  and the value is the data to append.
    """
    for key, value in kwargs.items():
        list_key = f"{key}_list"
        # Most keys follow the '_list' suffix convention
        if list_key in data_dict:
            data_dict[list_key].append(value)
        # Handle keys that might not follow the convention (e.g., 'calib' in some dicts)
        elif key in data_dict:
             data_dict[key].append(value)

def estimate_augmented_object_center(calib, original_obj, pc_in_frustum, class_name):
    """
    Estimates a plausible center for an augmented (impostor/pseudo) object
    using the point cloud in the original object's frustum.
    
    Args:
        calib: Calibration object for the scene.
        original_obj: The original kitti_object instance.
        pc_in_frustum: The point cloud extracted from the original object's frustum.
        class_name: The class name of the object ('Car', 'Pedestrian', etc.).
        
    Returns:
        A numpy array representing the estimated 3D center, or None if estimation fails.
    """
    estimated_depth = geometry.estimate_depth_from_box(calib.P, original_obj.box2d, original_obj.h)
    if estimated_depth <= 2:  # Ignore objects that are too close
        return None

    fake_range = FAKE_OBJECT_RANGE[class_name]
    app_pos = APP_POS[class_name]

    pts_in_frustum_T = pc_in_frustum.T[:3]
    if pts_in_frustum_T.shape[1] == 0:
        return None

    # Filter points by depth to find a plausible segment for the new object
    depth_mask = (pts_in_frustum_T[2] < (estimated_depth + fake_range[0])) & \
                 (pts_in_frustum_T[2] > (estimated_depth - fake_range[1]))

    if np.sum(depth_mask) < 5:
        return None

    seg_points = pts_in_frustum_T[:, depth_mask]
    return np.round(np.mean(seg_points, axis=1) + app_pos, 2)

def generate_impostor(scene_data: SceneData, original_obj, pc_in_frustum_from_original_obj):
    """
    Generates a synthetic 'impostor' object near a real object's location
    to be used as a challenging negative sample during training.
    
    Args:
        scene_data: The SceneData object for the current frame.
        original_obj: The original kitti_object instance.
        pc_in_frustum_from_original_obj: Point cloud from the original object's frustum.
        
    Returns:
        A dictionary containing all data for the generated impostor, or None if generation fails.
    """
    class_name = original_obj.type
    center = estimate_augmented_object_center(scene_data.calib, original_obj, pc_in_frustum_from_original_obj, class_name)
    if center is None:
        return None

    size = ObjectPriors.generate_proxy_size(class_name)
    
    # Find the best heading angle by minimizing the 2D box difference from the original
    best_heading, min_diff = 0, 1e7
    for i in range(12):
        heading = (i / 12.0) * np.pi
        pts3d = geometry.in_camera_coordinate(center, size, heading)
        pts2d = geometry.project_to_image(pts3d, scene_data.calib.P)
        box2d = np.array([min(pts2d[0]), min(pts2d[1]), max(pts2d[0]), max(pts2d[1])])
        diff = np.sum(np.abs(box2d - original_obj.box2d))
        if diff < min_diff:
            min_diff, best_heading = diff, heading

    heading = best_heading + np.random.uniform(-np.pi / 24, np.pi / 24)

    # Create the impostor's 3D and 2D bounding boxes
    box3d_3d = geometry.in_camera_coordinate(center, size, heading).T
    pts_2d = geometry.project_to_image(box3d_3d.T, scene_data.calib.P)
    box2d = np.array([np.min(pts_2d[0]), np.min(pts_2d[1]), np.max(pts_2d[0]), np.max(pts_2d[1])])
    
    img_height, img_width, _ = scene_data.img_shape
    box2d[[0, 2]] = np.clip(box2d[[0, 2]], 0, img_width - 1)
    box2d[[1, 3]] = np.clip(box2d[[1, 3]], 0, img_height - 1)
    
    # Extract point cloud and labels for the new impostor frustum
    pc_in_frustum, frustum_angle = geometry.extract_pc_in_frustum(
        scene_data.pc_rect, scene_data.pc_image_coord, scene_data.img_fov_inds, box2d, scene_data.calib
    )
    
    if len(pc_in_frustum) < 5:
        return None
        
    _, inds = geometry.extract_pc_in_box3d(pc_in_frustum, box3d_3d)
    labels = np.zeros(len(pc_in_frustum), dtype=np.int32)
    labels[inds] = 1

    return {
        'box2d': box2d, 'box3d': box3d_3d, 'points': pc_in_frustum.astype(np.float32),
        'label': labels, 'type': f'Impostor_{class_name}', 'heading': heading,
        'size': size, 'frustum_angle': frustum_angle
    }
