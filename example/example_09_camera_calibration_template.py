import os
import pickle
import numpy as np
from b3d import b3d

def make_intrinsic(width, height, fx=900.0, fy=920.0, cx=None, cy=None):
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def make_random_rotation(rng):
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q

    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def make_extrinsic(rng, center_range=3.0):
    """Create a fully random OpenGL-style world-to-camera matrix."""
    camera_pos = rng.uniform(-center_range, center_range, size=3)
    R = make_random_rotation(rng)
    t = -R @ camera_pos

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T.astype(np.float32)


# Build a small scene with calibration keys.
rng = np.random.default_rng()

width = np.random.randint(10, 1600)
height = np.random.randint(10, 1600)
fx = rng.uniform(1.0, 1500.0)
fy = rng.uniform(1.0, 1500.0)
K = make_intrinsic(width=width, height=height, fx=950.0, fy=900.0)
T = make_extrinsic(rng)


point_cloud = rng.random((20000, 3), dtype=np.float32) * 2.0 - 1.0
axes = np.eye(4, dtype=np.float32)


print('T:', T)
print('K:', K)

image = rng.random((int(height), int(width), 3)) * 255
image = image.astype(np.uint8)

template_data = {

    "pointcloud_#00A2FFDD_&3": point_cloud,
    "axis_&5": axes,
    'camera_1':{
        'intrinsic': K,
        'extrinsic': T, # Optional, can be identity if not provided. OpenGL-style world-to-camera matrix.
        'image': image, # Optional, can be used for visualization or ignored for calibration.
        # 'resolution': np.array([height, width], dtype=np.float32), # Optional, can be inferred from image shape.
        # 'depth': 10, # Optional. Default is 2.0.

    }
}

b3d.loadObj(template_data)