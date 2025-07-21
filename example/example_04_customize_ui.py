from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QDialog
import numpy as np
from qfluentwidgets import PushButton, Slider, ComboBox, SpinBox, LineEdit, MessageBox, Dialog
from PySide6.QtCore import Qt
import trimesh


pointslist = np.random.rand(100, 10, 1, 7)
lineslist = np.random.rand(100, 10, 2, 7)
bboxslist = np.random.rand(100, 10, 8, 7)
pointslist2 = np.random.rand(100, 10, 1, 3)
lineslist2 = np.random.rand(100, 10, 2, 3)
bboxslist2 = np.random.rand(100, 10, 8, 3)
# Build a simple GUI with a button and a slider

window = QWidget()
window.setWindowTitle('Test Window')
layout = QVBoxLayout(window)
button = PushButton('random_transform')
button2 = PushButton('resetCamera')
button3 = PushButton('translateCamera')
slider = Slider()
slider.setRange(0, 20)
slider.setValue(0)
slider.setOrientation(Qt.Horizontal)
layout.addWidget(button)
layout.addWidget(button2)
layout.addWidget(button3)
layout.addWidget(slider)
window.resize(300, 200)


# define user functions
def switch_objects(value):
    # NOTE: if you want to display lines and bboxs, 'line' or 'bbox' should be in the key
    #       and the shape of the data should be (..., 2, 3/6/7) or (..., 8, 3/6/7) respectively
    Batch3D.addObj({
        'points':pointslist[value],
        'lines':lineslist[value],
        'bboxs':bboxslist[value],
        'points2':pointslist2[value],
        'lines2':lineslist2[value],
        'bboxs2':bboxslist2[value]
    })

    

def random_transform():
    transform = trimesh.transformations.rotation_matrix(np.random.randn()*0.1, [0, 1, 0], [0, 0, 0])
    Batch3D.setObjTransform('bboxs', transform)

def resetCamera():
    Batch3D.GL.resetCamera()

def translateCamera():
    Batch3D.GL.camera.translateTo(10, 10, 10, True)


# connect the signals
button.clicked.connect(random_transform)
button2.clicked.connect(resetCamera)
button3.clicked.connect(translateCamera)
slider.valueChanged.connect(switch_objects)


transform = trimesh.transformations.rotation_matrix(np.random.randn()*0.1, [0, 1, 0], [0, 0, 0])
# show the window
window.show()
