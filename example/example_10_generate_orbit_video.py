import io
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QByteArray, QBuffer, QIODevice
from PySide6.QtWidgets import QApplication

from b3d import b3d


# Output and orbit settings. The script is executed from the example folder by
# Batch3D, so Path.cwd() is normally .../Batch3D/example.
# Use "gif" for broad compatibility, or "webp" for RGBA animation.
OUTPUT_FORMAT = "webp"
OUTPUT_PATH = Path.cwd() / f"camera_orbit_origin.{OUTPUT_FORMAT.lower()}"
FRAME_COUNT = 72
FRAME_DURATION_MS = 55

LOOK_AT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
DISTANCE = 4.0
ELEVATION = 25.0
START_AZIMUTH = 0.0
ORBIT_DEGREES = 360.0
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Set to None to keep the full OpenGL framebuffer size.
OUTPUT_WIDTH = 720


def ensure_demo_scene():
    """Add a small origin object when the current scene has no user objects."""
    if b3d.GL.getObjectList():
        return

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.75)
    sphere.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.array([70, 135, 255, 255], dtype=np.float32) / 255.0,
        metallicFactor=0.0,
        roughnessFactor=0.65,
    )

    b3d.add(
        {
            "origin_sphere": sphere,
            "axis_&4": np.eye(4, dtype=np.float32),
        }
    )


def process_events(repeat=2):
    app = QApplication.instance()
    if app is None:
        return

    for _ in range(repeat):
        app.processEvents()


def qimage_to_pil(image, keep_alpha=False):
    png_data = QByteArray()
    buffer = QBuffer(png_data)
    if not buffer.open(QIODevice.WriteOnly):
        raise RuntimeError("Failed to open QImage buffer")

    try:
        if not image.save(buffer, "PNG"):
            raise RuntimeError("Failed to encode QImage as PNG")
    finally:
        buffer.close()

    mode = "RGBA" if keep_alpha else "RGB"
    return Image.open(io.BytesIO(bytes(png_data))).convert(mode)


def capture_frame(keep_alpha=False):
    glw = b3d.GL

    # Force the GL widget to draw before grabbing the framebuffer. This is still
    # a blocking script, but processing events gives Qt/OpenGL a chance to paint.
    glw.update()
    process_events()
    glw.repaint()
    process_events()

    if hasattr(glw, "_grabRGBAMapImage"):
        qimage = glw._grabRGBAMapImage()
    else:
        qimage = glw.grabFramebuffer()

    frame = qimage_to_pil(qimage, keep_alpha=keep_alpha)

    if OUTPUT_WIDTH is not None and frame.width > OUTPUT_WIDTH:
        output_height = max(1, int(round(frame.height * OUTPUT_WIDTH / frame.width)))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        frame = frame.resize((OUTPUT_WIDTH, output_height), resample)

    return frame


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def make_look_at_transform(eye, target, up):
    """Build an OpenGL-style world-to-camera matrix looking at target."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z_axis = normalize(eye - target)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), z_axis)
    x_axis = normalize(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    transform = np.eye(4, dtype=np.float64)
    transform[0, :3] = x_axis
    transform[1, :3] = y_axis
    transform[2, :3] = z_axis
    transform[:3, 3] = -transform[:3, :3] @ eye
    return transform.astype(np.float32)


def set_camera(azimuth):
    camera = b3d.GL.camera
    azimuth_rad = np.deg2rad(float(azimuth))
    elevation_rad = np.deg2rad(float(ELEVATION))

    eye = LOOK_AT.astype(np.float64) + DISTANCE * np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=np.float64,
    )

    transform = make_look_at_transform(eye, LOOK_AT, WORLD_UP)
    camera.setCameraTransform(transform, isEmit=False, isAnimated=False)
    camera.updateProjTransform(isAnimated=False, isEmit=False)
    b3d.GL.update()


def save_frames(frames, output_format, output_path):
    output_format = output_format.lower()

    if output_format == "gif":
        frames = [frame.convert("RGB") for frame in frames]
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
            optimize=True,
        )
    elif output_format == "webp":
        frames = [frame.convert("RGBA") for frame in frames]
        frames[0].save(
            output_path,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
            lossless=True,
            quality=100,
            method=6,
        )
    else:
        raise ValueError(f'Unsupported OUTPUT_FORMAT: {output_format}. Use "gif" or "webp".')


def save_orbit_animation():
    output_format = OUTPUT_FORMAT.lower()
    keep_alpha = output_format == "webp"
    ensure_demo_scene()
    process_events(4)

    frames = []
    for i in range(FRAME_COUNT):
        t = i / FRAME_COUNT
        azimuth = START_AZIMUTH + ORBIT_DEGREES * t
        set_camera(azimuth)

        process_events()

        frames.append(capture_frame(keep_alpha=keep_alpha))
        print(f"Captured frame {i + 1}/{FRAME_COUNT}")

    if not frames:
        raise RuntimeError("No frames captured")

    save_frames(frames, output_format, OUTPUT_PATH)

    print(f"Saved camera orbit {output_format.upper()}: {OUTPUT_PATH}")


save_orbit_animation()
