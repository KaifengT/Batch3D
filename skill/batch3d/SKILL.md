---
name: batch3d
description: "Create Batch3D/b3d-compatible visualization artifacts only when the user explicitly mentions Batch3D or b3d for previewing, visualizing, saving a Batch3D-viewable file, or writing a Batch3D/b3d executable script. Use the save-file route when the user asks to save a visualization file or write a script that saves a visualization file. Use the API-script route when the user asks to use the Batch3D/b3d API, make an executable Batch3D/b3d script, interact with b3d.add/addObj/updateObj/rm, camera controls, UI, or other runtime APIs. Do not use for generic 3D visualization, mesh export, pickle/numpy saving, camera math, or point-cloud tasks unless Batch3D or b3d is explicitly mentioned. Includes a bundled b3d.pyi API reference for exact Batch3D script APIs. Also use for Chinese requests such as \u5e2e\u6211\u4f7f\u7528b3d\u53ef\u89c6\u5316."
---

# Batch3D Visualization

## Overview

Batch3D is an offline 3D viewer for local or remote 3D data. Produce either a saved dictionary file, usually `.pkl`, that Batch3D can load directly, or a Python script that runs in Batch3D and calls the `b3d` API.

This skill bundles `references/b3d.pyi`, copied from Batch3D's root API stub. Use this markdown file for visualization data schemas, and use `references/b3d.pyi` for exact callable APIs.

This skill also bundles `scripts/validate_batch3d_input.py`, a CLI validator for Batch3D-viewable saved files. Use it for final validation of generated `.pkl` files.

## Trigger Boundary

Use this skill only when the user explicitly mentions `Batch3D`, `batch3d`, or `b3d` and asks to preview, visualize, save a Batch3D-viewable artifact, or write a Batch3D/b3d script.

Do not use this skill for generic requests about point clouds, meshes, pickle files, numpy files, camera/image visualization preview, 3D visualization, or rendering unless the user ties the task to Batch3D/b3d. In those generic cases, answer with the normal project or Python context instead.

## Bundled API Reference

Read `references/b3d.pyi` before writing or modifying an API-script route artifact that calls `b3d`, `b3d.GL`, camera methods, object-update methods, UI-related methods, signals, or low-level object classes. It contains the current public API signatures for `b3d`, `GLWidget`, `GLCamera`, `PointCloud`, `Lines`, `BoundingBox`, `Mesh`, and related classes.

Do not read `references/b3d.pyi` for a plain save-file route unless the task also needs runtime API calls. Do not rely on memory for method names or parameters when a task asks for script behavior beyond the examples below. Resolve the reference path relative to this skill directory.

## Bundled Validation Script

Run `scripts/validate_batch3d_input.py` after creating or modifying a Batch3D saved visualization file. The script takes one saved file path, usually a `.pkl`, and prints the validation result to the CLI.

Resolve the script path relative to this skill directory:

```bash
python scripts/validate_batch3d_input.py path/to/scene.pkl
```

Use the CLI output as the source of truth. Do not replace it with manual validation.

## Route Selection

Choose one primary route. Do not mix the save-file route and API-script route unless the user explicitly asks for both a saved visualization artifact and a runtime Batch3D/b3d script.

Save-file route:
- Use when the user mentions saving a file, saving a visualization file, creating a Batch3D-viewable `.pkl/.npy/.npz`, or writing a script whose job is to save a visualization file.
- Create a top-level `dict[str, value]` and write it with `pickle.dump(..., "wb")` by default.
- Prefer the schemas in this file. Do not call `b3d` or require Batch3D to be running unless the user asks for runtime preview.

API-script route:
- Use when the user asks to use Batch3D/b3d APIs, write a script to run inside Batch3D, interactively preview/update objects, use `b3d.add`, `b3d.updateObj`, `b3d.rm`, `b3d.GL`, camera controls, UI widgets, signals, or object properties.
- Read `references/b3d.pyi` before using APIs beyond the minimal examples.
- Import `from b3d import b3d` and call the runtime API directly.

## Workflow

1. Select the save-file route or API-script route using the rules above.

2. Use supported values:
   - `numpy.ndarray` for point clouds, lines, bounding boxes, coordinate transforms, and axes.
   - `trimesh.Trimesh` or `trimesh.Scene` objects for mesh data by default, including meshes with PBR materials, transparency, or textures.
   - `dict` with `vertex` and `face` for mesh data only when the user explicitly asks for a vertex/face dictionary or a pure numpy mesh representation.
   - `dict` whose key contains `camera` for camera rendering visualization/preview and optional image rendering visualization/preview.
   - File paths only inside scripts, not as the default saved pickle format.

3. Normalize arrays:
   - Use finite numeric arrays, preferably `np.float32` for geometry and `np.uint8` for RGB images.
   - Use 0..1 float color channels for embedded geometry colors.
   - For explicit vertex/face mesh dictionaries, use integer mesh faces with shape `(M, 3)`.

## Key Syntax

Use descriptive object keys. Batch3D also parses key substrings and suffixes:

- Prefer plain keys without color or size suffixes, for example `points`, `trajectory_line`, `object_bbox`, `mesh_cube`.
- Add lowercase `line` in the key for line segments. Example: `trajectory_line`.
- Add lowercase `bbox` in the key for bounding boxes. Example: `object_bbox`.
- Add `camera` in the key for camera/image visualization preview dictionaries. Camera matching is case-insensitive.
- Add `#RRGGBB` or `#RRGGBBAA` only when the user explicitly requests a uniform color. If adding size too, put an underscore after the hex code: `points_#00A2FFDD_&3`.
- Add `&<number>` only when the user explicitly requests point or line size. Example: `axis_&5`, `pointcloud_&10`.

If geometry arrays include per-vertex colors, embedded colors override the key color.

## Data Schema

Point cloud:
- Key: any key that is not parsed as `line`, `bbox`, or `camera`.
- Value: ndarray shaped `(..., N, 3)`, `(..., N, 6)`, or `(..., N, 7)`.
- Columns: `x, y, z`; optional `r, g, b`; optional `a`.

Line segments:
- Key: must contain lowercase `line`.
- Value: ndarray shaped `(..., 2, 3)`, `(..., 2, 6)`, or `(..., 2, 7)`.
- The second-to-last dimension stores `[start, end]`.

Bounding boxes:
- Key: must contain lowercase `bbox`.
- Value: ndarray shaped `(..., 8, 3)`, `(..., 8, 6)`, or `(..., 8, 7)`.
- Use this vertex order for an axis-aligned box:

```python
[
    [xmin, ymin, zmax], [xmin, ymax, zmax],
    [xmax, ymax, zmax], [xmax, ymin, zmax],
    [xmin, ymin, zmin], [xmin, ymax, zmin],
    [xmax, ymax, zmin], [xmax, ymin, zmin],
]
```

Coordinate transforms or axes:
- Key: any descriptive key, commonly `axis` or `pose_axis`. Use `axis_&5` only when the user asks for a thick displayed axis line.
- Value: ndarray shaped `(..., 4, 4)`.
- Batch3D renders each homogeneous transform as RGB coordinate axes.
- If user ask for a larger axis size, multiply the transform's rotation part by the size factor. For example, for a 5x larger axis, multiply the first 3 columns of the `(4, 4)` transform by `5.0`.

Mesh:
- Key: not preferred, mainly depend on object class.
- Preferred value: a `trimesh.Trimesh` or `trimesh.Scene` object. Batch3D can load trimesh objects from pickle entries, and this preserves `trimesh` materials, PBR settings, transparency, UVs, and textures.
- For PBR or transparent mesh visualization, set `mesh.visual.material = trimesh.visual.material.PBRMaterial(...)`. For alpha blending, use `alphaMode="BLEND"` and an RGBA `baseColorFactor`.
- Fallback value, only when explicitly requested: dict with `vertex` and `face`.
- Fallback `vertex`: ndarray shaped `(N, 3)`, `(N, 6)`, `(N, 7)`, `(N, 9)`, or `(N, 10)`.
- Fallback vertex columns: `xyz`, optional `rgb` or `rgba`, optional normal `nx, ny, nz` for 9/10-column vertices.
- Fallback `face`: integer ndarray shaped `(M, 3)` with valid vertex indices.

Camera/Image Visualization Preview:
- Key: contains `camera`, for example `camera_main`.
- Value: dict with:
  - `intrinsic` or `intrinsics`: required `(3, 3)` camera matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`; `fx` and `fy` must be positive.
  - `extrinsic` or `extrinsics`: optional `(4, 4)` OpenGL-style world-to-camera matrix. The last row must be `[0, 0, 0, 1]`, the rotation must be orthonormal, and the camera looks along local `-Z`. Identity means the camera is at the world origin looking toward `-Z`.
  - `resolution`: optional `(2,)` array `[height, width]`; no batched camera resolution is supported.
  - `image`: optional `(H, W, 3)` RGB image, preferably `np.uint8`; Batch3D projects it as a textured background plane.
  - `depth`: optional scalar projection depth for the image plane; default is `2.0`. calculate depth to place the image plane behind the nearest geometry, or use a default of `5.0` if geometry depth is not known.


Do not batch camera dictionaries. Store each camera as a separate dict entry if multiple cameras are needed.

Camera coordinate systems:
- Batch3D previews cameras in OpenGL camera space: `+X` right, `+Y` up, and the camera looks along local `-Z` (`+Z` points backward).
- Before saving a camera preview, infer the user's source camera convention from framework names, variable names, documentation, or request text. Common sources are PyTorch3D and OpenCV. If the source convention is ambiguous and the preview matters, ask for clarification.
- If the user's external matrix `T_src_w2c` maps source-world points into source-camera coordinates, convert it for Batch3D with `T_b3d_w2c = T_src_camera_to_opengl_camera @ T_src_w2c`.
- If mesh/world coordinates are also converted into a different world basis, apply the matching world conversion consistently to mesh, points, poses, and camera extrinsics. If mesh and camera are both left in the original source world, only pre-multiply the camera extrinsic by the source-camera-to-OpenGL-camera matrix.


Use these matrices directly:

```python
import numpy as np


def pytorch3d_to_opengl_transform() -> np.ndarray:
    """
    Convert PyTorch3D camera coordinates to OpenGL camera coordinates.

    PyTorch3D camera: X left, Y up, Z forward.
    OpenGL camera: X right, Y up, Z backward; camera looks along -Z.
    This is a 180 degree rotation around the Y axis.
    """
    return np.array([
        [-1.0, 0.0,  0.0, 0.0],
        [ 0.0, 1.0,  0.0, 0.0],
        [ 0.0, 0.0, -1.0, 0.0],
        [ 0.0, 0.0,  0.0, 1.0],
    ], dtype=np.float32)


def opencv_to_opengl_transform() -> np.ndarray:
    """
    Convert OpenCV camera coordinates to OpenGL camera coordinates.

    OpenCV camera: X right, Y down, Z forward.
    OpenGL camera: X right, Y up, Z backward; camera looks along -Z.
    This is a 180 degree rotation around the X axis.
    """
    return np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ], dtype=np.float32)
```

Examples:

```python
# PyTorch3D mesh and camera, both kept in the PyTorch3D world basis.
T_p3d_w2c = np.eye(4, dtype=np.float32)
camera_entry = {
    "intrinsics": K,
    "extrinsics": pytorch3d_to_opengl_transform() @ T_p3d_w2c,
    "resolution": np.array([height, width], dtype=np.float32),
}

# OpenCV world-to-camera extrinsic.
T_cv_w2c = np.eye(4, dtype=np.float32)
camera_entry = {
    "intrinsics": K,
    "extrinsics": opencv_to_opengl_transform() @ T_cv_w2c,
    "resolution": np.array([height, width], dtype=np.float32),
}
```

## Batches

Batch3D slices arrays with more than 2 dimensions by the first dimension. The default slice value `-1` loads the whole array and flattens all leading dimensions for display. Slice values `0..T-1` load `array[t:t+1]`.

Use these patterns:

- Time sequence of point clouds: `(T, N, 3)` or `(T, N, 6/7)`.
- Time sequence of lines: `(T, L, 2, 3/6/7)`.
- Time sequence of boxes: `(T, B, 8, 3/6/7)`.
- Multiple poses: `(T, 4, 4)` or `(B, T, 4, 4)` if first-dimension slicing is desired.

## Saved Pickle Pattern

Use this shape as the default for user requests like "save a file I can view in Batch3D":

```python
import pickle
import numpy as np
import trimesh


def make_bbox(xmin, ymin, zmin, xmax, ymax, zmax):
    return np.array([
        [xmin, ymin, zmax], [xmin, ymax, zmax],
        [xmax, ymax, zmax], [xmax, ymin, zmax],
        [xmin, ymin, zmin], [xmin, ymax, zmin],
        [xmax, ymax, zmin], [xmax, ymin, zmin],
    ], dtype=np.float32)


def make_pbr_cube():
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    cube.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.array([176, 108, 108, 180], dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=0.82,
        alphaMode="BLEND",
        doubleSided=True,
    )
    return cube


def pytorch3d_to_opengl_transform() -> np.ndarray:
    return np.array([
        [-1.0, 0.0,  0.0, 0.0],
        [ 0.0, 1.0,  0.0, 0.0],
        [ 0.0, 0.0, -1.0, 0.0],
        [ 0.0, 0.0,  0.0, 1.0],
    ], dtype=np.float32)


def opencv_to_opengl_transform() -> np.ndarray:
    return np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ], dtype=np.float32)


points = np.random.rand(2000, 3).astype(np.float32)
line_segments = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)
box = make_bbox(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5)[None]
pose = np.eye(4, dtype=np.float32)[None]
mesh = make_pbr_cube()

height, width = 720, 1280
K = np.array([
    [900.0, 0.0, width / 2.0],
    [0.0, 900.0, height / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

save_dt = {
    "points": points,
    "trajectory_line": line_segments,
    "object_bbox": box,
    "pose_axis": pose,
    "mesh_cube": mesh,
    "camera_main": {
        "intrinsic": K,
        # Use np.eye(4) only for an OpenGL-style camera.
        # For PyTorch3D/OpenCV camera previews, pre-multiply the source
        # world-to-camera extrinsic by the matching conversion matrix.
        "extrinsic": np.eye(4, dtype=np.float32),
        "resolution": np.array([height, width], dtype=np.float32),
        "depth": 4.0,
    },
}

with open("scene.pkl", "wb") as f:
    pickle.dump(save_dt, f)
```

If the user explicitly asks for a vertex/face mesh dictionary, use `{"vertex": vertices, "face": faces}` instead of a trimesh object.

## Script Pattern

Use this shape for user requests like "write a visualization script":

```python
from b3d import b3d
import numpy as np
import trimesh


points = np.random.rand(2000, 3).astype(np.float32)
transform = np.eye(4, dtype=np.float32)
transform[:3, 3] = [0.5, 0.0, 0.0]

cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
cube.visual.material = trimesh.visual.material.PBRMaterial(
    baseColorFactor=np.array([176, 108, 108, 180], dtype=np.uint8),
    metallicFactor=0.0,
    roughnessFactor=0.82,
    alphaMode="BLEND",
    doubleSided=True,
)

b3d.add({
    "points": points,
    "axis": np.eye(4, dtype=np.float32),
    "mesh_cube": cube,
})

b3d.setObjTransform("points", transform)
b3d.GL.camera.setCamera(distance=5.0)
```

Useful script calls:

- `b3d.add(data)` or `b3d.addObj(data)`: merge objects into the current scene.
- `b3d.updateObj(data)`: reset the current scene with new objects.
- `b3d.rm(key_or_keys)` or `b3d.rmObj(key_or_keys)`: remove objects.
- `b3d.clear()`: clear all objects.
- `b3d.getWorkspaceObj()`: get the current raw workspace dictionary.
- `b3d.setObjTransform(name, transform)`: set an object's `(4, 4)` transform.
- `b3d.setObjectProps(name, {"size": value, "isShow": bool})`: update object display properties.
- `b3d.add("path/to/model.obj")`: load supported mesh files such as `.obj`, `.ply`, `.stl`, `.glb`, `.gltf`, `.pcd`, or `.xyz`.

If creating a PySide UI script, keep the widget in a module-level variable and call `window.show()` so the window is not garbage-collected.

## Final Checks

For the save-file route, do not perform extra manual checks. After producing the `.pkl`, run the bundled validator directly:

```bash
python scripts/validate_batch3d_input.py path/to/scene.pkl
```

Base the next step only on the validator's CLI output:

- `OK`: finish and report the saved file path.
- `OK_WITH_WARNINGS`: finish if the warnings are acceptable for the user's request; otherwise modify the file and rerun the validator.
- `INVALID` or any command failure: modify the saved file or generation script, then rerun the validator before finishing.

For an API-script route that does not create a saved `.pkl`, skip this validator and rely on the relevant runtime/API checks requested by the user.
