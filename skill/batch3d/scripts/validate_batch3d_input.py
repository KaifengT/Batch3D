#!/usr/bin/env python
"""Validate .pkl/.npz/.npy files against Batch3D display input rules."""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None  # type: ignore


SUPPORTED_EXTENSIONS = {".pkl", ".npz", ".npy"}
ARRAY_TAIL_DIMS = (3, 6, 7)
MESH_VERTEX_DIMS = (3, 6, 7, 9, 10)


def install_numpy_pickle_aliases() -> None:
    """Allow NumPy 2 pickles to load in NumPy 1.x environments when possible."""
    try:
        import numpy.core as numpy_core
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "numeric", "fromnumeric", "umath", "_multiarray_umath"):
        try:
            module = __import__(f"numpy.core.{name}", fromlist=[name])
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


@dataclass
class Issue:
    level: str
    key: str | None
    reason: str


@dataclass
class ValidationResult:
    errors: list[Issue]
    warnings: list[Issue]
    checked_keys: int = 0
    ignored_keys: int = 0

    def add_error(self, key: str | None, reason: str) -> None:
        self.errors.append(Issue("ERROR", key, reason))

    def add_warning(self, key: str | None, reason: str) -> None:
        self.warnings.append(Issue("WARNING", key, reason))


def load_batch3d_file(path: Path) -> dict[Any, Any]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported file extension {path.suffix!r}; expected .pkl, .npz, or .npy"
        )

    if ext == ".pkl":
        install_numpy_pickle_aliases()
        with path.open("rb") as f:
            obj = pickle.load(f)
    elif ext == ".npz":
        with np.load(path, allow_pickle=True) as obj:
            obj = {key: obj[key] for key in obj.files}
    else:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, dict):
            pass
        elif isinstance(obj, np.lib.npyio.NpzFile):
            obj = dict(obj)
        elif isinstance(obj, np.ndarray):
            obj = {"numpy file": obj}
        else:
            raise ValueError(f"unknown numpy file type: {type(obj).__name__}")

    if not isinstance(obj, dict):
        raise ValueError(f"top-level object must be a dict, got {type(obj).__name__}")
    return obj


def is_trimesh_value(value: Any) -> bool:
    if trimesh is None:
        return False

    classes: list[type] = []
    parent = getattr(trimesh, "parent", None)
    geometry3d = getattr(parent, "Geometry3D", None)
    for cls in (
        geometry3d,
        getattr(trimesh, "Scene", None),
        getattr(trimesh, "Trimesh", None),
        getattr(trimesh, "PointCloud", None),
    ):
        if isinstance(cls, type):
            classes.append(cls)
    return bool(classes) and isinstance(value, tuple(classes))


def validate_numeric_array(
    key: str,
    arr: np.ndarray,
    result: ValidationResult,
    *,
    context: str = "ndarray",
    warn_nonfinite: bool = True,
) -> bool:
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr.astype(np.float32)
        except (TypeError, ValueError):
            result.add_error(
                key,
                f"{context} dtype {arr.dtype} cannot be converted to float32 geometry data",
            )
            return False
        result.add_warning(
            key,
            f"{context} dtype {arr.dtype} is not numeric; Batch3D will cast it to float32",
        )

    if warn_nonfinite:
        try:
            if not np.isfinite(arr.astype(np.float64, copy=False)).all():
                result.add_warning(
                    key,
                    f"{context} contains NaN or Inf values; Batch3D will replace them with np.nan_to_num",
                )
        except (TypeError, ValueError, OverflowError):
            result.add_error(key, f"{context} contains values that cannot be checked as finite")
            return False
    return True


def validate_array_key_suffixes(key: str, result: ValidationResult) -> None:
    if "#" in key:
        hex_part = key.split("#", 2)[1].split("_", 2)[0]
        if len(hex_part) in (6, 8):
            try:
                int(hex_part, 16)
            except ValueError:
                result.add_error(
                    key,
                    f"color suffix '#{hex_part}' is not valid hexadecimal; Batch3D will fail while decoding it",
                )
        else:
            result.add_warning(
                key,
                f"color suffix '#{hex_part}' is ignored because it is not #RRGGBB or #RRGGBBAA",
            )

    if "&" in key:
        size_part = key.split("&", 2)[1]
        try:
            float(size_part)
        except ValueError:
            result.add_warning(
                key,
                f"size suffix '&{size_part}' is not a number; Batch3D will use the default size",
            )


def validate_array(key: str, value: Any, result: ValidationResult) -> None:
    result.checked_keys += 1
    arr = np.asarray(value)
    shape = arr.shape

    validate_array_key_suffixes(key, result)
    numeric_ok = validate_numeric_array(key, arr, result)

    is_line_shape = len(shape) >= 2 and shape[-2] == 2 and shape[-1] in ARRAY_TAIL_DIMS
    is_bbox_shape = len(shape) >= 2 and shape[-2] == 8 and shape[-1] in ARRAY_TAIL_DIMS
    is_axis_shape = len(shape) >= 2 and shape[-2] == 4 and shape[-1] == 4
    is_point_shape = len(shape) >= 2 and shape[-1] in ARRAY_TAIL_DIMS

    has_line = "line" in key
    has_bbox = "bbox" in key
    has_line_casefold = "line" in key.lower()
    has_bbox_casefold = "bbox" in key.lower()

    if is_line_shape and not has_line:
        if has_line_casefold:
            result.add_warning(
                key,
                f"shape {shape} looks like line segments, but Batch3D requires lowercase 'line' in the key",
            )
        else:
            result.add_warning(
                key,
                f"shape {shape} looks like line segments, but the key does not contain lowercase 'line'; Batch3D will display it as a point cloud",
            )

    if is_bbox_shape and not has_bbox:
        if has_bbox_casefold:
            result.add_warning(
                key,
                f"shape {shape} looks like bounding boxes, but Batch3D requires lowercase 'bbox' in the key",
            )
        else:
            result.add_warning(
                key,
                f"shape {shape} looks like bounding boxes, but the key does not contain lowercase 'bbox'; Batch3D will display it as a point cloud",
            )

    if has_line and not is_line_shape:
        result.add_warning(
            key,
            f"key contains 'line' but shape {shape} is not (..., 2, 3/6/7); Batch3D will not display it as line segments",
        )

    if has_bbox and not is_bbox_shape:
        result.add_warning(
            key,
            f"key contains 'bbox' but shape {shape} is not (..., 8, 3/6/7); Batch3D will not display it as bounding boxes",
        )

    if not numeric_ok:
        return

    if is_line_shape and has_line:
        return
    if is_bbox_shape and has_bbox:
        return
    if is_point_shape:
        return
    if is_axis_shape:
        return

    result.add_error(
        key,
        f"ndarray shape {shape} is not supported; expected point cloud (..., N, 3/6/7), line (..., 2, 3/6/7) with lowercase 'line', bbox (..., 8, 3/6/7) with lowercase 'bbox', or transform (..., 4, 4)",
    )


def match_camera_key(key: Any) -> str | None:
    norm_key = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    if "intrinsic" in norm_key:
        return "intrinsic"
    if "extrinsic" in norm_key:
        return "extrinsic"
    if "resolution" in norm_key:
        return "resolution"
    if "depth" in norm_key:
        return "depth"
    return None


def normalize_calibration_matrix(key: str, value: Any, mode: str, result: ValidationResult) -> bool:
    arr = np.asarray(value)
    if mode == "resolution":
        if arr.shape != (2,):
            result.add_error(key, f"camera resolution must be shape (2,), got {arr.shape}")
            return False
        try:
            res = arr.astype(np.float64)
        except (TypeError, ValueError):
            result.add_error(key, "camera resolution cannot be converted to float64")
            return False
        if not np.isfinite(res).all():
            result.add_error(key, "camera resolution contains non-finite values")
            return False
        if np.any(res <= 0):
            result.add_error(key, f"camera resolution must be positive, got {res.tolist()}")
            return False
        return True

    expected = (3, 3) if mode == "intrinsic" else (4, 4)
    if arr.shape != expected:
        result.add_error(key, f"camera {mode} matrix must be shape {expected}, got {arr.shape}")
        return False

    try:
        mat = arr.astype(np.float64)
    except (TypeError, ValueError):
        result.add_error(key, f"camera {mode} matrix cannot be converted to float64")
        return False

    if not np.isfinite(mat).all():
        result.add_error(key, f"camera {mode} matrix contains non-finite values")
        return False

    if mode == "intrinsic":
        if not np.isclose(mat[2, 2], 1.0, atol=1e-6):
            result.add_error(key, "intrinsic matrix K[2,2] must be 1.0")
            return False
        zero_entries = [mat[0, 1], mat[1, 0], mat[2, 0], mat[2, 1]]
        if not np.allclose(zero_entries, 0.0, atol=1e-8):
            result.add_error(
                key,
                "intrinsic matrix must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]",
            )
            return False
        if mat[0, 0] <= 0 or mat[1, 1] <= 0:
            result.add_error(key, "intrinsic matrix fx and fy must be positive")
            return False
        return True

    if not np.allclose(mat[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        result.add_error(key, "extrinsic matrix last row must be [0, 0, 0, 1]")
        return False

    rot = mat[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4):
        result.add_error(key, "extrinsic rotation must be orthonormal")
        return False

    det = float(np.linalg.det(rot))
    if not np.isclose(det, 1.0, atol=1e-4):
        result.add_error(key, f"extrinsic rotation determinant must be 1.0, got {det:.6g}")
        return False
    return True


def validate_camera_dict(key: str, value: dict[Any, Any], result: ValidationResult) -> None:
    result.checked_keys += 1
    intrinsic = None
    extrinsic = None
    resolution = None
    image = None

    for sub_key, sub_value in value.items():
        mode = match_camera_key(sub_key)
        if mode == "intrinsic":
            intrinsic = sub_value
        elif mode == "extrinsic":
            extrinsic = sub_value
        elif mode == "resolution":
            resolution = sub_value
        elif str(sub_key).lower() == "image":
            image = sub_value
        elif mode == "depth":
            if not isinstance(sub_value, (int, float)):
                result.add_warning(
                    key,
                    f"camera depth field {sub_key!r} is not a Python int/float; Batch3D will use default depth 2.0",
                )

    if intrinsic is None:
        result.add_error(key, "camera dict is missing required intrinsic/intrinsics matrix")
    else:
        normalize_calibration_matrix(key, intrinsic, "intrinsic", result)

    if extrinsic is not None:
        normalize_calibration_matrix(key, extrinsic, "extrinsic", result)

    if resolution is not None:
        normalize_calibration_matrix(key, resolution, "resolution", result)

    if image is not None:
        image_arr = np.asarray(image)
        if not (image_arr.ndim == 3 and image_arr.shape[-1] == 3):
            result.add_error(key, f"camera image must be shape (H, W, 3), got {image_arr.shape}")
        elif not np.issubdtype(image_arr.dtype, np.number):
            result.add_error(key, f"camera image dtype {image_arr.dtype} must be numeric")
        else:
            if image_arr.dtype != np.uint8:
                result.add_warning(
                    key,
                    f"camera image dtype is {image_arr.dtype}; np.uint8 RGB is recommended",
                )
            if not np.isfinite(image_arr.astype(np.float64, copy=False)).all():
                result.add_error(key, "camera image contains non-finite values")


def validate_mesh_dict(key: str, value: dict[Any, Any], result: ValidationResult) -> None:
    result.checked_keys += 1
    if "vertex" not in value:
        result.add_error(key, "mesh dict is missing required key 'vertex'")
    if "face" not in value:
        result.add_error(key, "mesh dict is missing required key 'face'")
    if "vertex" not in value or "face" not in value:
        return

    vertex = value["vertex"]
    face = value["face"]

    if not hasattr(vertex, "shape"):
        result.add_error(key, "mesh vertex must be a numpy array")
        vertex_arr = None
    else:
        vertex_arr = np.asarray(vertex)
        if vertex_arr.ndim != 2 or vertex_arr.shape[-1] not in MESH_VERTEX_DIMS:
            result.add_error(
                key,
                f"mesh vertex shape must be (N, 3), (N, 6), (N, 7), (N, 9), or (N, 10), got {vertex_arr.shape}",
            )
        validate_numeric_array(key, vertex_arr, result, context="mesh vertex")

    if not hasattr(face, "shape"):
        result.add_error(key, "mesh face must be a numpy array")
        return

    face_arr = np.asarray(face)
    if face_arr.ndim != 2 or face_arr.shape[-1] != 3:
        result.add_error(key, f"mesh face shape must be (M, 3), got {face_arr.shape}")
        return

    if not np.issubdtype(face_arr.dtype, np.integer):
        result.add_error(key, f"mesh face dtype must be integer, got {face_arr.dtype}")
        return

    if face_arr.size:
        if np.any(face_arr < 0):
            result.add_error(key, "mesh face indices must be non-negative")
        if vertex_arr is not None and vertex_arr.ndim >= 1:
            vertex_count = int(vertex_arr.shape[0])
            if vertex_count == 0:
                result.add_error(key, "mesh vertex array is empty but faces are present")
            elif np.any(face_arr >= vertex_count):
                result.add_error(
                    key,
                    f"mesh face indices must be less than vertex count {vertex_count}",
                )


def validate_trimesh_value(key: str, value: Any, result: ValidationResult) -> None:
    result.checked_keys += 1
    if trimesh is None:
        result.add_warning(key, "trimesh is not installed; trimesh object validation was skipped")
        return

    scene_cls = getattr(trimesh, "Scene", None)
    mesh_cls = getattr(trimesh, "Trimesh", None)
    pointcloud_cls = getattr(trimesh, "PointCloud", None)

    if isinstance(scene_cls, type) and isinstance(value, scene_cls):
        if not getattr(value, "geometry", None):
            result.add_error(key, "trimesh.Scene contains no geometry")
        return

    if isinstance(mesh_cls, type) and isinstance(value, mesh_cls):
        vertices = np.asarray(value.vertices)
        faces = np.asarray(value.faces)
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            result.add_error(key, f"trimesh vertices must be shape (N, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[-1] != 3:
            result.add_error(key, f"trimesh faces must be shape (M, 3), got {faces.shape}")
        if vertices.size == 0:
            result.add_warning(key, "trimesh has no vertices")
        if faces.size == 0:
            result.add_warning(key, "trimesh has no faces")
        return

    if isinstance(pointcloud_cls, type) and isinstance(value, pointcloud_cls):
        vertices = np.asarray(value.vertices)
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            result.add_error(key, f"trimesh.PointCloud vertices must be shape (N, 3), got {vertices.shape}")
        return

    result.add_error(
        key,
        f"unsupported trimesh geometry type {value.__class__.__name__}; expected Trimesh, Scene, or PointCloud",
    )


def validate_object(data: dict[Any, Any]) -> ValidationResult:
    result = ValidationResult(errors=[], warnings=[])
    for raw_key, value in data.items():
        key = str(raw_key)

        if isinstance(value, dict):
            if "camera" in key.lower():
                validate_camera_dict(key, value, result)
            else:
                validate_mesh_dict(key, value, result)
        elif is_trimesh_value(value):
            validate_trimesh_value(key, value, result)
        elif hasattr(value, "shape"):
            validate_array(key, value, result)
        else:
            result.ignored_keys += 1

    return result


def print_report(path: Path, result: ValidationResult) -> None:
    if result.errors:
        status = "INVALID"
    elif result.warnings:
        status = "OK_WITH_WARNINGS"
    else:
        status = "OK"

    print(
        f"{status}: {path} checked_keys={result.checked_keys} ignored_keys={result.ignored_keys} "
        f"errors={len(result.errors)} warnings={len(result.warnings)}"
    )

    for issue in result.errors + result.warnings:
        if issue.key is None:
            print(f"{issue.level}: {issue.reason}")
        else:
            print(f"{issue.level} key={issue.key!r}: {issue.reason}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Batch3D-compatible .pkl, .npz, or .npy input file."
    )
    parser.add_argument("input", type=Path, help="Path to a .pkl, .npz, or .npy file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    path = args.input

    try:
        data = load_batch3d_file(path)
    except Exception as exc:
        result = ValidationResult(errors=[Issue("ERROR", None, f"failed to load input: {exc}")], warnings=[])
        print_report(path, result)
        return 2

    result = validate_object(data)
    print_report(path, result)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
