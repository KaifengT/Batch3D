from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from qfluentwidgets import PushButton, BodyLabel, PrimaryPushButton
import numpy as np

class DistanceMeasureWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.point1 = None
        self.point2 = None  # 顶点
        self.point3 = None
        self.click_count = 0
        self.mode = 'distance'  # 'distance' or 'angle'
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('点间距离与夹角测量')
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title_label = BodyLabel('点间距离与夹角测量工具')
        layout.addWidget(title_label)
        
        description_label = BodyLabel(
            '使用方法：\n'
            '1. 点击“开始测量距离”或“开始测量夹角”按钮\n'
            '2. 在主窗口中依次点击点（使用鼠标中键）\n'
            '3. 显示结果，并在主界面中可视化'
        )
        layout.addWidget(description_label)
        
        self.status_label = BodyLabel('状态：等待开始')
        layout.addWidget(self.status_label)
        
        self.result_label = BodyLabel('结果：未计算')
        layout.addWidget(self.result_label)
        
        button_layout = QHBoxLayout()
        self.start_distance_button = PushButton('开始测量距离')
        self.start_distance_button.clicked.connect(self.start_distance_measurement)
        
        self.start_angle_button = PushButton('开始测量夹角')
        self.start_angle_button.clicked.connect(self.start_angle_measurement)
        
        self.reset_button = PushButton('重置')
        self.reset_button.clicked.connect(self.reset_measurement)
        
        button_layout.addWidget(self.start_distance_button)
        button_layout.addWidget(self.start_angle_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)
        
        # 连接外部信号
        b3d.GL.middleMouseClickSignal.connect(self.on_point_selected)

    def start_distance_measurement(self):
        self.mode = 'distance'
        self.click_count = 0
        self.point1 = None
        self.point2 = None
        self.status_label.setText('状态：请在主窗口中点击第一个点')
        self.result_label.setText('距离：未计算')

    def start_angle_measurement(self):
        self.mode = 'angle'
        self.click_count = 0
        self.point1 = None
        self.point2 = None  # 顶点
        self.point3 = None
        self.status_label.setText('状态：请在主窗口中点击第一个点（角的一边）')
        self.result_label.setText('夹角：未计算')

    def reset_measurement(self):
        self.click_count = 0
        self.point1 = None
        self.point2 = None
        self.point3 = None
        self.status_label.setText('状态：等待开始')
        self.result_label.setText('结果：未计算')
        b3d.rm(['point1', 'point2', 'point3', 'line1', 'line2'])

    def on_point_selected(self, UV, P3D):
        point = np.array(P3D[:3])
        
        if self.mode == 'distance':
            self.handle_distance_click(point)
        elif self.mode == 'angle':
            self.handle_angle_click(point)

    def handle_distance_click(self, point):
        if self.click_count >= 2:
            return

        if self.click_count == 0:
            self.point1 = point
            self.click_count = 1
            self.status_label.setText('状态：已选择第一个点，请点击第二个点')
            self.add_point_to_scene(point, name='point1')
        elif self.click_count == 1:
            self.point2 = point
            self.click_count = 2
            self.calculate_distance()
            self.status_label.setText('状态：已完成两点选择')
            self.add_point_to_scene(point, name='point2')
            self.draw_line_between_points('point1', 'point2', 'line1')

    def handle_angle_click(self, point):
        if self.click_count >= 3:
            return

        if self.click_count == 0:
            self.point1 = point
            self.click_count = 1
            self.status_label.setText('状态：已选择第一个点，请点击顶点')
            self.add_point_to_scene(point, name='point1')
        elif self.click_count == 1:
            self.point2 = point  # 顶点
            self.click_count = 2
            self.status_label.setText('状态：已选择顶点，请点击第三个点')
            self.add_point_to_scene(point, name='point2')
        elif self.click_count == 2:
            self.point3 = point
            self.click_count = 3
            self.calculate_angle()
            self.status_label.setText('状态：已完成三点选择')
            self.add_point_to_scene(point, name='point3')
            self.draw_line_between_points('point1', 'point2', 'line1')
            self.draw_line_between_points('point2', 'point3', 'line2')

    def add_point_to_scene(self, point, name='point'):
        point_cloud = point[None, :]  # shape: (1, 3)
        b3d.add({name: point_cloud})
        b3d.setObjectProps(name, {'size': 20})

    def draw_line_between_points(self, p1_name, p2_name, line_name):
        p1 = getattr(self, p1_name.replace('point', 'point'))
        p2 = getattr(self, p2_name.replace('point', 'point'))
        if p1 is not None and p2 is not None:
            line_points = np.stack([p1, p2], axis=0)
            b3d.add({line_name: line_points})
            b3d.setObjectProps(line_name, {'size': 20})

    def calculate_distance(self):
        if self.point1 is not None and self.point2 is not None:
            dist = np.linalg.norm(self.point2 - self.point1)
            self.result_label.setText(f'距离：{dist:.3f}')
            b3d.popMessage(title='距离计算完成', message=f'测量的距离为 {dist:.3f}', mtype='msg', followMouse=True)
        else:
            self.result_label.setText('距离：无法计算')

    def calculate_angle(self):
        if self.point1 is not None and self.point2 is not None and self.point3 is not None:
            # 向量 a = point1 -> point2
            # 向量 b = point3 -> point2
            a = self.point1 - self.point2
            b = self.point3 - self.point2
            
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止浮点误差
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.rad2deg(angle_rad)
            
            self.result_label.setText(f'夹角：{angle_deg:.2f}°')
            b3d.popMessage(title='夹角计算完成', message=f'测量的夹角为 {angle_deg:.2f}°', mtype='msg')
        else:
            self.result_label.setText('夹角：无法计算')
            
window = DistanceMeasureWidget()
window.show()