import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QSize, QUrl
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import (
    QMediaCaptureSession,
    QMediaFormat,
    QMediaRecorder,
    QVideoFrame,
    QVideoFrameInput,
)
from PySide6.QtWidgets import QApplication

from b3d import b3d


# Output and orbit settings. The script is executed from the example folder by
# Batch3D, so Path.cwd() is normally .../Batch3D/example.
# Use "gif" for broad compatibility, "webp" for RGBA animation,
# or "mp4" for QtMultimedia H.264/MPEG-4 video.
OUTPUT_FORMAT = "mp4"
OUTPUT_PATH = Path.cwd() / f"camera_orbit_origin.{OUTPUT_FORMAT.lower()}"
FRAME_COUNT = 160
FRAME_DURATION_MS = 16
MP4_FRAME_SEND_TIMEOUT_SEC = 5.0
MP4_STOP_TIMEOUT_SEC = 10.0

LOOK_AT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
DISTANCE = 4.0
ELEVATION = 25.0
START_AZIMUTH = 0.0
ORBIT_DEGREES = 360.0
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Set to None to keep the full OpenGL framebuffer size.
OUTPUT_WIDTH = 3840

FRAME_STAGE_ORDER = (
    "set_camera",
    "repaint",
    "screenshot",
    "qimage_to_pil",
    "resize",
)

FRAME_STAGE_LABELS = {
    "set_camera": "set_camera",
    "repaint": "repaint()",
    "screenshot": "grab framebuffer",
    "qimage_to_pil": "QImage -> PIL",
    "resize": "resize",
}


def add_timing(stats, key, elapsed):
    if stats is None:
        return
    stats[key] = stats.get(key, 0.0) + elapsed


def timed_call(stats, key, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    add_timing(stats, key, time.perf_counter() - start)
    return result


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
    rgba_image = image.convertToFormat(QImage.Format_RGBA8888)
    width = rgba_image.width()
    height = rgba_image.height()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot convert empty QImage: {width}x{height}")

    bytes_per_line = rgba_image.bytesPerLine()
    row_bytes = width * 4
    raw = np.frombuffer(rgba_image.constBits(), dtype=np.uint8, count=rgba_image.sizeInBytes())

    if bytes_per_line == row_bytes:
        rgba_bytes = raw[: height * row_bytes].tobytes()
    else:
        rows = raw.reshape((height, bytes_per_line))[:, :row_bytes]
        rgba_bytes = np.ascontiguousarray(rows).tobytes()

    frame = Image.frombytes("RGBA", (width, height), rgba_bytes, "raw", "RGBA")
    return frame if keep_alpha else frame.convert("RGB")


def capture_frame(keep_alpha=False, stats=None):
    glw = b3d.GL

    timed_call(stats, "repaint", glw.repaint)

    if hasattr(glw, "_grabRGBAMapImage"):
        qimage = timed_call(stats, "screenshot", glw._grabRGBAMapImage)
    else:
        qimage = timed_call(stats, "screenshot", glw.grabFramebuffer)

    frame = timed_call(stats, "qimage_to_pil", qimage_to_pil, qimage, keep_alpha=keep_alpha)
    return resize_frame(frame, stats=stats)


def resize_frame(frame, stats=None):
    resize_start = time.perf_counter()
    if OUTPUT_WIDTH is not None and frame.width > OUTPUT_WIDTH:
        output_height = max(1, int(round(frame.height * OUTPUT_WIDTH / frame.width)))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        frame = frame.resize((OUTPUT_WIDTH, output_height), resample)
    add_timing(stats, "resize", time.perf_counter() - resize_start)

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
            method=0,
        )
    elif output_format == "mp4":
        save_frames_mp4(frames, output_path)
    else:
        raise ValueError(f'Unsupported OUTPUT_FORMAT: {output_format}. Use "gif", "webp", or "mp4".')


def select_mp4_media_format():
    media_format = QMediaFormat()
    supported_formats = media_format.supportedFileFormats(QMediaFormat.ConversionMode.Encode)
    if QMediaFormat.FileFormat.MPEG4 not in supported_formats:
        names = ", ".join(fmt.name for fmt in supported_formats)
        raise RuntimeError(f"QtMultimedia cannot encode MPEG4/MP4 on this system. Supported: {names}")

    media_format.setFileFormat(QMediaFormat.FileFormat.MPEG4)
    supported_codecs = media_format.supportedVideoCodecs(QMediaFormat.ConversionMode.Encode)
    for codec in (QMediaFormat.VideoCodec.H264, QMediaFormat.VideoCodec.MPEG4):
        if codec in supported_codecs:
            media_format.setVideoCodec(codec)
            return media_format

    names = ", ".join(codec.name for codec in supported_codecs)
    raise RuntimeError(f"QtMultimedia cannot encode H.264 or MPEG-4 video on this system. Supported: {names}")


def pil_frame_to_qvideo_frame(frame, index, fps, frame_duration_us):
    rgb_frame = frame.convert("RGB")
    width, height = rgb_frame.size
    if width % 2 or height % 2:
        rgb_frame = rgb_frame.crop((0, 0, width - (width % 2), height - (height % 2)))
        width, height = rgb_frame.size

    rgb_bytes = rgb_frame.tobytes()
    image = QImage(rgb_bytes, width, height, width * 3, QImage.Format_RGB888)
    image = image.convertToFormat(QImage.Format_RGB32)

    video_frame = QVideoFrame(image)
    video_frame.setStartTime(index * frame_duration_us)
    video_frame.setEndTime((index + 1) * frame_duration_us)
    video_frame.setStreamFrameRate(float(fps))
    return video_frame


def wait_for_recorder_state(app, recorder, target_state, timeout_sec):
    deadline = time.perf_counter() + timeout_sec
    while recorder.recorderState() != target_state and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)

    return recorder.recorderState() == target_state


def send_video_frame(app, video_input, video_frame, timeout_sec):
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        if video_input.sendVideoFrame(video_frame):
            return
        app.processEvents()
        time.sleep(0.005)

    raise TimeoutError("Timed out waiting for QtMultimedia to accept an MP4 video frame")


def save_frames_mp4(frames, output_path):
    if not frames:
        raise RuntimeError("No frames to save")

    app = QApplication.instance()
    if app is None:
        raise RuntimeError("MP4 export requires a running QApplication")

    output_path = Path(output_path)
    first_frame = frames[0].convert("RGB")
    width, height = first_frame.size
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid MP4 frame size: {first_frame.size}")

    fps = 1000.0 / float(FRAME_DURATION_MS)
    frame_duration_us = int(round(FRAME_DURATION_MS * 1000.0))

    session = QMediaCaptureSession()
    recorder = QMediaRecorder()
    video_input = QVideoFrameInput()

    errors = []
    recorder.errorOccurred.connect(lambda error, message: errors.append((error, message)))

    recorder.setMediaFormat(select_mp4_media_format())
    recorder.setVideoFrameRate(float(fps))
    recorder.setVideoResolution(QSize(width, height))
    recorder.setQuality(QMediaRecorder.Quality.HighQuality)
    recorder.setOutputLocation(QUrl.fromLocalFile(str(output_path.resolve())))

    session.setRecorder(recorder)
    session.setVideoFrameInput(video_input)

    recorder.record()
    if not wait_for_recorder_state(app, recorder, QMediaRecorder.RecorderState.RecordingState, MP4_STOP_TIMEOUT_SEC):
        raise RuntimeError(f"QtMultimedia did not start MP4 recording: {recorder.errorString()}")

    try:
        for index, frame in enumerate(frames):
            if frame.size != first_frame.size:
                frame = frame.resize(first_frame.size, getattr(Image, "Resampling", Image).LANCZOS)
            video_frame = pil_frame_to_qvideo_frame(frame, index, fps, frame_duration_us)
            send_video_frame(app, video_input, video_frame, MP4_FRAME_SEND_TIMEOUT_SEC)
    finally:
        recorder.stop()
        wait_for_recorder_state(app, recorder, QMediaRecorder.RecorderState.StoppedState, MP4_STOP_TIMEOUT_SEC)

    if errors:
        message = "; ".join(str(message) for _, message in errors if message)
        raise RuntimeError(f"QtMultimedia MP4 export failed: {message or errors[-1][0]}")

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"QtMultimedia MP4 export produced an empty file: {output_path}")


def print_timing_summary(frame_timings, save_elapsed):
    if not frame_timings:
        return

    frame_total = sum(frame.get("frame_total", 0.0) for frame in frame_timings)
    overall_total = frame_total + save_elapsed
    frame_count = len(frame_timings)

    print("")
    print("Orbit video timing summary")
    print(f"Frames: {frame_count}")
    print(f"Frame loop total: {frame_total:.3f}s")
    print(f"Save frames total: {save_elapsed:.3f}s")
    print(f"Overall measured total: {overall_total:.3f}s")
    if overall_total > 0.0:
        print(f"Save share of measured total: {save_elapsed / overall_total * 100.0:5.1f}%")
    print("")
    print(f"{'Stage':<30} {'Avg ms':>10} {'Max ms':>10} {'Frame %':>9}")
    print("-" * 64)

    for key in FRAME_STAGE_ORDER:
        values = [frame.get(key, 0.0) for frame in frame_timings]
        stage_total = sum(values)
        avg_ms = stage_total / frame_count * 1000.0
        max_ms = max(values) * 1000.0
        share = (stage_total / frame_total * 100.0) if frame_total > 0.0 else 0.0
        print(f"{FRAME_STAGE_LABELS[key]:<30} {avg_ms:10.2f} {max_ms:10.2f} {share:8.1f}%")

    frame_totals = [frame.get("frame_total", 0.0) for frame in frame_timings]
    print("-" * 64)
    print(
        f"{'frame total':<30} "
        f"{sum(frame_totals) / frame_count * 1000.0:10.2f} "
        f"{max(frame_totals) * 1000.0:10.2f} "
        f"{100.0:8.1f}%"
    )
    print("")


def save_orbit_animation():
    output_format = OUTPUT_FORMAT.lower()
    keep_alpha = output_format == "webp"
    ensure_demo_scene()
    process_events(4)

    frames = []
    frame_timings = []
    for i in range(FRAME_COUNT):
        frame_stats = {}
        frame_start = time.perf_counter()
        t = i / FRAME_COUNT
        azimuth = START_AZIMUTH + ORBIT_DEGREES * t
        timed_call(frame_stats, "set_camera", set_camera, azimuth)

        frames.append(capture_frame(keep_alpha=keep_alpha, stats=frame_stats))
        frame_stats["frame_total"] = time.perf_counter() - frame_start
        frame_timings.append(frame_stats)
        # print(f"Captured frame {i + 1}/{FRAME_COUNT} ({frame_stats['frame_total'] * 1000.0:.1f} ms)")

    if not frames:
        raise RuntimeError("No frames captured")

    save_start = time.perf_counter()
    save_frames(frames, output_format, OUTPUT_PATH)
    save_elapsed = time.perf_counter() - save_start

    print(f"Saved camera orbit {output_format.upper()}: {OUTPUT_PATH}")
    print_timing_summary(frame_timings, save_elapsed)


save_orbit_animation()
