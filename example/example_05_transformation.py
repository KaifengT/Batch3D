from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from PySide6.QtGui import QWheelEvent
import numpy as np
from qfluentwidgets import PushButton, ComboBox, PrimaryPushButton, BodyLabel
import pickle
from scipy.spatial.transform import Rotation as R


# ==============================
# 自定义支持滚轮的标签控件
# ==============================
class WheelLabel(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0.0
        self.step = 0.1
        self.min_val = -1000
        self.max_val = 1000
        self.is_rotation = False

    def set_as_rotation(self):
        """设置为旋转模式"""
        self.is_rotation = True
        self.min_val = -360
        self.max_val = 360
        self.step = 0.2

    def set_as_translation(self):
        """设置为平移模式"""
        self.is_rotation = False
        self.min_val = -1000
        self.max_val = 1000
        self.step = 0.1

    def wheelEvent(self, event: QWheelEvent):
        """处理鼠标滚轮事件"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.value = min(self.value + self.step, self.max_val)
        else:
            self.value = max(self.value - self.step, self.min_val)
        # 调用全局变换处理器
        if 'transform_processor' in globals():
            transform_processor.process_transform()

    def setValue(self, value):
        """设置当前值"""
        self.value = value

    def getValue(self):
        """获取当前值"""
        return self.value


# ==============================
# 变换管理器
# ==============================
class TransformManager:
    def __init__(self):
        self.accumulated_transforms = {}

    def reset_transform(self, name):
        self.accumulated_transforms[name] = np.eye(4)

    def apply_incremental_transform(self, name, dr, dp, dy, dtx, dty, dtz):
        if name not in self.accumulated_transforms:
            self.reset_transform(name)

        rotation = R.from_euler('xyz', [dr, dp, dy], degrees=True)
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[0, 3] = dtx
        transform[1, 3] = dty
        transform[2, 3] = dtz

        self.accumulated_transforms[name] = self.accumulated_transforms[name] @ transform
        print(f"Applied transform to {name}: R({dr:.1f}°, {dp:.1f}°, {dy:.1f}°), T({dtx:.1f}, {dty:.1f}, {dtz:.1f})")
        return self.accumulated_transforms[name]

    def save_transforms(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.accumulated_transforms, f)


# ==============================
# 变换处理器
# ==============================
class TransformProcessor:
    def __init__(self, wheel_labels, target_widget, transform_manager):
        self.wheel_labels = wheel_labels
        self.target_widget = target_widget
        self.transform_manager = transform_manager

    def process_transform(self):
        """处理变换逻辑"""
        if len(self.wheel_labels) >= 6:
            values = [label.getValue() for label in self.wheel_labels]
            r, p, y, tx, ty, tz = values

            if any(values):
                print(f'Incremental transform: R({r}, {p}, {y}), T({tx}, {ty}, {tz})')
                target = self.target_widget.currentText()
                if target == "all":
                    for name in b3d.GL.objectList.keys():
                        self.transform_point(r, p, y, tx, ty, tz, name)
                elif target and 'b3d' in globals():
                    self.transform_point(r, p, y, tx, ty, tz, target)
                
            if any(values):
                self.clear_inputs()

    def transform_point(self, r, p, y, tx, ty, tz, name):
        transform = self.transform_manager.apply_incremental_transform(name, r, p, y, tx, ty, tz)
        if 'b3d' in globals():
            b3d.setObjTransform(name, transform)
        else:
            print(f"transform {name}")

    def clear_inputs(self):
        """清空输入值"""
        for label in self.wheel_labels:
            label.setValue(0.0)


# ==============================
# 界面控制器
# ==============================
class UIController:
    def __init__(self, target_widget, res_widget, transform_manager, transform_processor):
        self.target_widget = target_widget
        self.res_widget = res_widget
        self.transform_manager = transform_manager
        self.transform_processor = transform_processor

    def get_workspace(self):
        objlist = []
        if 'b3d' in globals() and hasattr(b3d, 'GL'):
            objlist = list(b3d.GL.objectList.keys())
            if objlist:
                objlist.append("all")
        self.target_widget.clear()
        self.target_widget.addItems(objlist)
        print('objlist:', objlist)
        return objlist

    def reset_current_transform(self):
        target = self.target_widget.currentText()
        if target == "all":
            for name in b3d.GL.objectList.keys():
                self.transform_manager.reset_transform(name)
                if 'b3d' in globals():
                    b3d.setObjTransform(name, np.eye(4))
        elif target:
            self.transform_manager.reset_transform(target)
            if 'b3d' in globals():
                b3d.setObjTransform(target, np.eye(4))

    def save_transform(self):
        r1, r2 = QFileDialog.getSaveFileName(
            None, 'Save Transform', '', 'Transform Files (*.pkl)')
        print('save file:', r1, r2)
        if not r1:
            return
        self.transform_manager.save_transforms(r1)
        print('saved:', self.transform_manager.accumulated_transforms)

    def set_resolution(self, res):
        print('set resolution:', res)
        step = float(res)
        # 设置旋转控件步长
        for i in range(3):
            if i < len(wheel_labels):
                wheel_labels[i].step = step * 2
        # 设置平移控件步长
        for i in range(3, 6):
            if i < len(wheel_labels):
                wheel_labels[i].step = step


# ==============================
# 全局变量
# ==============================
wheel_labels = []
target_name_widget = None
res_combobox = None
transform_manager = TransformManager()
transform_processor = None
ui_controller = None


# ==============================
# 创建主窗口（保持与原代码完全一致）
# ==============================
window = QWidget()
window.setWindowTitle('Transformer')
window.setStyleSheet("""
    QWidget {
        font-family: 'Microsoft YaHei';
        font-size: 12px;
    }
""")

layout = QVBoxLayout(window)
h1layout = QHBoxLayout()
h2layout = QHBoxLayout()
layout.addLayout(h1layout)
layout.addLayout(h2layout)

# 创建按钮
button = PrimaryPushButton('💾 Save Transform')
button2 = PushButton('🔄 Refresh Objects')
button_reset = PushButton('↺ Reset Current Transform')

# 创建下拉框
target_name_widget = ComboBox()
res_combobox = ComboBox()
res_combobox.addItems(['0.01', '0.1', '0.2', '1.0', '10.0'])

# 创建支持滚轮的标签
label1 = ['Roll', 'Pitch', 'Yaw']
label2 = ['X', 'Y', 'Z']

# 创建旋转标签
for i in range(3):
    label = WheelLabel(label1[i])
    label.set_as_rotation()
    wheel_labels.append(label)
    h1layout.addWidget(label)
    
# 创建平移标签
for i in range(3):
    label = WheelLabel(label2[i])
    label.set_as_translation()
    wheel_labels.append(label)
    h2layout.addWidget(label)

# 初始化控制器
transform_processor = TransformProcessor(wheel_labels, target_name_widget, transform_manager)
ui_controller = UIController(target_name_widget, res_combobox, transform_manager, transform_processor)

# 连接按钮信号
button.clicked.connect(ui_controller.save_transform)
button2.clicked.connect(ui_controller.get_workspace)
button_reset.clicked.connect(ui_controller.reset_current_transform)
res_combobox.currentTextChanged.connect(ui_controller.set_resolution)

# 添加其他控件
layout.addWidget(button2)
layout.addWidget(BodyLabel('🎯 Target Object:'))
layout.addWidget(target_name_widget)
layout.addWidget(BodyLabel('⚙️ Set Resolution:'))
layout.addWidget(res_combobox)
layout.addSpacing(10)
layout.addWidget(button_reset)
layout.addWidget(button)

# 设置布局间距
layout.setSpacing(8)
h1layout.setSpacing(6)
h2layout.setSpacing(6)

# 初始化工作区
objlist = ui_controller.get_workspace()
print('objlist:', objlist)

window.resize(350, 320)
window.show()