import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from qfluentwidgets import PushButton, PrimaryPushButton, BodyLabel, ComboBox, DoubleSpinBox
from collections.abc import Mapping

PARTS: Mapping = Batch3D.getWorkspaceObj()
print(f"获取到的部件: {list(PARTS.keys())}")
def getParts(data:dict):
    PARTS = data
    print(f"获取到的部件: {list(PARTS.keys())}")


Batch3D.workspaceUpdatedSignal.connect(getParts)


'''
PARTS['2500shiploader.gantry'] 大车
PARTS['2500shiploader.Turntable'] 回转台
PARTS['2500shiploader.Boom'] 大梁
PARTS['2500shiploader.TailCar'] 尾车
PARTS['2500shiploader.Chute0'] 溜筒0
PARTS['2500shiploader.Chute1'] 溜筒1
PARTS['2500shiploader.Chute2'] 溜筒2
PARTS['2500shiploader.Chute3'] 溜筒3
'''

Batch3D.setObjTransform('2500shiploader.gantry', np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))


class controlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slider Widget")
        self.setGeometry(100, 100, 300, 100)

        self.gantry_spinbox = DoubleSpinBox(self)
        self.gantry_spinbox.setRange(-180, 180)
        self.gantry_spinbox.setValue(0)


        layout = QHBoxLayout(self)
        layout.addWidget(BodyLabel('大车'))
        layout.addWidget(self.gantry_spinbox)


        self.gantry_spinbox.valueChanged.connect(self.on_slider_value_changed)
        
        self.setLayout(layout)

    def on_slider_value_changed(self, value: float):
        Batch3D.setObjTransform('2500shiploader.gantry', np.array([
            [1, 0, 0, value],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))


panel = controlPanel()
panel.show()