**[English](README.md), [Chinese](README_zh.md)**

# Batch3D
A tool for batch viewing local or remote 3D data.

![image](asset/cover1.png)

## Launch
### Install Dependencies
First, ensure all required dependencies are installed:
```bash
pip install -r requirements.txt
```

### Start the Program
There are two methods to launch Batch3D:
- Double-click the run.bat file;
- Run python Batch3D.py in the command line.

### Open Directory and File Loading
1. Click the "Local Folder" button or drag a folder/file into the window. Filenames will appear in the right-side list.
2. Click list items or use the keyboard's up/down arrows to switch files quickly.
3. Double-click a list item to reload the file.

### View Remote Server Files
1. Click the "Remote Folder" button and enter the server’s IP address, username, password, and other information.
2. Batch3D will connect to the server and download files for viewing.

## How to Save Viewable Data

### File Formats and Organization

Batch3D supports the following formats: `.pkl`, `.npy`, `.npz`, `.ply`, `.obj`, `.stl` etc.

1. `.pkl`, `.npy`, `.npz` files should store dictionary-type data in binary format. Values should use `numpy.ndarray` or `dict` classes. Example:

```Python
import pickle
import numpy as np
save_dt = {
    'pcd1_#00FF00': np.random.rand(100, 3),  # Point cloud
    'pcd2_#888888': np.random.rand(5, 100, 3),  # Batched point cloud
    'pcd3': np.random.rand(5, 100, 6), # Batched point cloud with point-wise rgb color
    'pcd4': np.random.rand(5, 1, 4, 100, 7) # High-dim batched point cloud with point-wise rgba color
    'line1_#123456': np.random.rand(5, 100, 2, 3),  # Line segments. NOTE: for data visualization as lines, 'line' must be in the keys.
    'bbox1_#123456': np.array([
        [[0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],]
        ]),  # Bounding box, NOTE: for data visualization as bounding boxes, 'bbox' must be in the keys.
        # Vertex order:
        #      Z
        #      |
        #      0-----1
        #     /|    /|
        #    3-----2 |
        #    | |   | |
        #    | 4---|-5 ---> Y
        #    |/    |/
        #    7-----6
        #   /
        #  X 
    'mesh': {
        'vertex': np.random.rand(233, 3), # or (N, 6), (N, 7)
        'face':   np.random.randint(0, 233, size=(514, 3)),
    }
}
with open("test.pkl", 'wb') as f:
    pickle.dump(save_dt, f)
```
Batch3D can then parse the `test.pkl` file.


### Data Dimensions
#### Dimension and Type Recognition

For `.pkl`, `.npy`, `.npz` files, Batch3D automatically determines display methods based on array dimensions and key identifiers:

- Point Cloud: ndarray with shape `(..., N, 3)`, `(..., N, 6)` or `(..., N, 7)`，where $N > 2$；;

- Line: ndarray with keys containing `line` and shape `(..., 2, 3)`, `(..., 2, 6)` or `(..., 2, 7)`;

- Bounding Box: ndarray with keys containing `bbox` and shape `(..., 8, 3)`, `(..., 8, 6)` or `(..., 8, 7)`;

- Homogeneous Transformation: ndarray with shape `(..., 4, 4)`;

- Mesh: dict containing two keys:

    1. vertex: ndarray with shape `(N, 3)`, `(N, 6)`, or `(N, 7)` (mesh vertex xyz, xyzrgb, or xyzrgba);

    2. face: ndarray with shape `(M, 3)` (mesh vertex indices, integer type).

Other data types are currently unsupported.

#### Batch Processing

- For `.pkl`, `.npy`, `.npz` files: Higher-dimensional data is treated as batch data for sliced display. When the slice index is set to `-1`, Batch3D merges all dimensions for display. For other slice values, Batch3D slices the first dimension while merging the remaining dimensions.

#### Color Specification
For `.pkl`, `.npy`, `.npz` files, append `#HHHHHH` or `#HHHHHHHH` hexadecimal color codes to keys to specify colors for point clouds, lines, or bounding boxes. If unspecified, colors are automatically assigned.

For point clouds, lines, bounding boxes and mesh vertices, colors can also be embedded as `(x, y, z, r, g, b)` or `(x, y, z, r, g, b, a)` .

### Custom Camera Parameters and Image Projection

Camera calibration can be stored as a dictionary whose key contains `camera`. The camera entry supports:

- `intrinsic`: required 3x3 camera intrinsic matrix `K`.
- `extrinsic`: optional 4x4 OpenGL-style world-to-camera matrix. The camera looks along local `-Z`.
- `resolution`: optional `(height, width)` array. This is used for the camera mask/output region.
- `image`: optional `(H, W, 3)` RGB image. If provided, it is projected into the 3D scene as a textured background plane.
- `depth`: optional image projection depth. Defaults to `2.0`.

Camera matrices are validated strictly. Batch-like camera intrinsics/extrinsics/resolution are not supported; use exactly `(3, 3)`, `(4, 4)`, and `(2,)`.

Simple reference file:

```Python
import pickle
import numpy as np

height, width = 720, 1280

K = np.array([
    [900.0, 0.0, width / 2.0],
    [0.0, 900.0, height / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

# OpenGL-style world-to-camera extrinsic.
# Identity means the camera is at the world origin and looks toward -Z.
T_world_to_camera = np.eye(4, dtype=np.float32)

points = np.random.rand(2000, 3).astype(np.float32)
points[:, :2] = points[:, :2] * 2.0 - 1.0
points[:, 2] = -np.random.uniform(1.0, 4.0, size=points.shape[0])

image = np.zeros((height, width, 3), dtype=np.uint8)
image[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
image[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
image[..., 2] = 128

save_dt = {
    "points_#00A2FFDD_&3": points,
    "camera_demo": {
        "intrinsic": K,
        "extrinsic": T_world_to_camera,
        "resolution": np.array([height, width], dtype=np.float32),
        "image": image,
        "depth": 4.0,
    },
}

with open("camera_projection_demo.pkl", "wb") as f:
    pickle.dump(save_dt, f)
```

## Run Scripts

Refer to example scripts in [example1](example\example_01_random_pcd.py), [example2](example\example_02_trimesh_obj.py), [example3](example\example_04_customize_ui.py), and [camera calibration example](example\example_09_camera_calibration_template.py).
