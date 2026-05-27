from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from qfluentwidgets import BodyLabel, Slider
import numpy as np
import trimesh

from b3d import b3d


OBJECT_SPECS = {
    "transparent_cube_red": {
        "rgb": np.array([176, 108, 108], dtype=np.uint8),
        "translation": np.array([-1.25, 0.0, 0.0], dtype=np.float32),
    },
    "transparent_cube_green": {
        "rgb": np.array([110, 165, 124], dtype=np.uint8),
        "translation": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    },
    "transparent_cube_blue": {
        "rgb": np.array([104, 126, 178], dtype=np.uint8),
        "translation": np.array([1.25, 0.0, 0.0], dtype=np.float32),
    },
}

INITIAL_OPACITY = 0.45
SLIDER_SCALE = 100


def make_pbr_material(rgb, opacity):
    opacity = float(np.clip(opacity, 0.0, 1.0))
    rgba = np.concatenate(
        [np.asarray(rgb, dtype=np.uint8), np.array([round(opacity * 255)], dtype=np.uint8)]
    )

    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=rgba,
        metallicFactor=0.0,
        roughnessFactor=0.82,
        alphaMode="BLEND",
        doubleSided=True,
    )


def make_cube(rgb, translation, opacity):
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    cube.visual.material = make_pbr_material(rgb, opacity)
    cube.apply_translation(translation)
    return cube


def build_scene(opacity):
    return {
        name: make_cube(spec["rgb"], spec["translation"], opacity)
        for name, spec in OBJECT_SPECS.items()
    }


def set_opacity(value):
    opacity = float(value) / SLIDER_SCALE
    opacity_value_label.setText(f"{opacity:.2f}")
    b3d.add(build_scene(opacity))


b3d.add(build_scene(INITIAL_OPACITY))

window = QWidget()
window.setWindowTitle("PBR Cube Opacity")

layout = QVBoxLayout(window)
layout.setContentsMargins(16, 14, 16, 14)
layout.setSpacing(10)

row = QHBoxLayout()
row.setSpacing(10)

opacity_label = BodyLabel("Opacity")
opacity_value_label = BodyLabel(f"{INITIAL_OPACITY:.2f}")
opacity_value_label.setMinimumWidth(42)
opacity_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

slider = Slider()
slider.setOrientation(Qt.Horizontal)
slider.setRange(0, SLIDER_SCALE)
slider.setValue(round(INITIAL_OPACITY * SLIDER_SCALE))
slider.valueChanged.connect(set_opacity)

row.addWidget(opacity_label)
row.addWidget(slider, 1)
row.addWidget(opacity_value_label)
layout.addLayout(row)

window.resize(320, 84)
window.show()
