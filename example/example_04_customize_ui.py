from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QDialog
import numpy as np
from qfluentwidgets import PushButton, Slider, ComboBox, SpinBox, LineEdit, MessageBox, Dialog
from PySide6.QtCore import Qt
import trimesh


datalist = np.random.rand(100, 500, 7)

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
    data =  datalist[value]
    Batch3D.addObj({'data':data})
    

def random_transform():
    transform = trimesh.transformations.rotation_matrix(np.random.randn()*0.1, [0, 1, 0], [0, 0, 0])
    Batch3D.setObjTransform('data', transform)

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
