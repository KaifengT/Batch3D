from PySide6.QtCore import Qt, QObject
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QSizePolicy,
                               QSpacerItem)
from PySide6.QtGui import QKeySequence
import webbrowser
from qfluentwidgets import (Action, BodyLabel, DropDownToolButton, SegmentedWidget, RoundMenu, \
                            SpinBox, DoubleSpinBox, SwitchButton, Slider)
from qfluentwidgets import FluentIcon as FIF





class GLSettingWidget(QObject):


    def __init__(self, parent=None, 
                 render_mode_callback=None, 
                 camera_control_callback=None, 
                 camera_persp_callback=None,
                 camera_view_callback=None,
                 reset_camera_callback=None, 
                 fov_callback=None,
                 far_callback=None,
                 near_callback=None,
                 grid_vis_callback=None,
                 axis_vis_callback=None,
                 axis_length_callback=None,
                 save_depth_callback=None,
                 save_rgba_callback=None,
                 enable_ssao_callback=None):
        super().__init__()
        
        self.parent = parent
        self.render_mode_callback = render_mode_callback
        self.camera_control_callback = camera_control_callback
        self.camera_persp_callback = camera_persp_callback
        self.camera_view_callback = camera_view_callback
        self.reset_camera_callback = reset_camera_callback
        self.fov_callback = fov_callback
        self.far_callback = far_callback
        self.near_callback = near_callback
        self.grid_vis_callback = grid_vis_callback
        self.axis_vis_callback = axis_vis_callback
        self.axis_length_callback = axis_length_callback
        self.save_depth_callback = save_depth_callback
        self.save_rgba_callback = save_rgba_callback
        self.enable_ssao_callback = enable_ssao_callback

        self._setup_ui()
        
    def _setup_ui(self):
        
        self.gl_setting_button = DropDownToolButton(FIF.SETTING, self.parent)
        
        self.gl_setting_Menu = RoundMenu(parent=self.parent)
        
        frame = QFrame()
        frame.setLayout(QVBoxLayout())
        frame.layout().setContentsMargins(0, 10, 0, 15)
        frame.layout().setSpacing(10)

        self.gl_render_mode_combobox = SegmentedWidget(parent=self.gl_setting_Menu)
        self.gl_render_mode_combobox.addItem('0', ' Line ', lambda: self._on_render_mode_changed(0))
        self.gl_render_mode_combobox.addItem('1', 'Simple', lambda: self._on_render_mode_changed(1))
        self.gl_render_mode_combobox.addItem('2', 'Normal', lambda: self._on_render_mode_changed(2))
        self.gl_render_mode_combobox.addItem('3', 'Texture', lambda: self._on_render_mode_changed(3))
        self.gl_render_mode_combobox.setCurrentItem('1')
        self.gl_render_mode_combobox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        
        self.gl_camera_control_combobox = SegmentedWidget(parent=self.gl_setting_Menu)
        self.gl_camera_control_combobox.addItem('0', 'Arcball', lambda: self._on_camera_control_changed(0))
        self.gl_camera_control_combobox.addItem('1', ' Orbit ', lambda: self._on_camera_control_changed(1))
        self.gl_camera_control_combobox.setCurrentItem('0')
        self.gl_camera_control_combobox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        
        self.gl_camera_perp_combobox = SegmentedWidget(parent=self.gl_setting_Menu)
        self.gl_camera_perp_combobox.addItem('0', 'Perspective', lambda: self._on_camera_persp_changed(0))
        self.gl_camera_perp_combobox.addItem('1', 'Orthographic', lambda: self._on_camera_persp_changed(1))
        self.gl_camera_perp_combobox.setCurrentItem('0')    
        self.gl_camera_perp_combobox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        
        
        self.gl_camera_view_combobox = SegmentedWidget(parent=self.gl_setting_Menu)
        self.gl_camera_view_combobox.addItem('0', '+X', lambda: self._on_camera_view_changed(0))
        self.gl_camera_view_combobox.addItem('1', '-X', lambda: self._on_camera_view_changed(1))
        self.gl_camera_view_combobox.addItem('2', '+Y', lambda: self._on_camera_view_changed(2))
        self.gl_camera_view_combobox.addItem('3', '-Y', lambda: self._on_camera_view_changed(3))
        self.gl_camera_view_combobox.addItem('4', '+Z', lambda: self._on_camera_view_changed(4))
        self.gl_camera_view_combobox.addItem('5', '-Z', lambda: self._on_camera_view_changed(5))
        self.gl_camera_view_combobox.addItem('6', 'Free', lambda: self._on_camera_view_changed(6))
        self.gl_camera_view_combobox.setCurrentItem('0')    
        self.gl_camera_view_combobox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        
        
        
        
        frame.layout().addWidget(BodyLabel("Render Mode", parent=self.gl_setting_Menu))
        frame.layout().addWidget(self.gl_render_mode_combobox)
        frame.layout().addWidget(BodyLabel("Camera Control", parent=self.gl_setting_Menu))
        frame.layout().addWidget(self.gl_camera_control_combobox)
        frame.layout().addWidget(BodyLabel("Camera Projection", parent=self.gl_setting_Menu))
        frame.layout().addWidget(self.gl_camera_perp_combobox)
        frame.layout().addWidget(BodyLabel("Camera View", parent=self.gl_setting_Menu))
        frame.layout().addWidget(self.gl_camera_view_combobox)
        frame.adjustSize()

        self.gl_setting_Menu.addWidget(frame, selectable=False)
        self.gl_setting_Menu.addSeparator()
        
        frame = QFrame()
        frame.setLayout(QHBoxLayout())
        frame.layout().setContentsMargins(0, 10, 0, 10)
        frame.layout().setSpacing(20)
        self.fov_spinbox = SpinBox(parent=self.gl_setting_Menu)
        self.fov_spinbox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        self.fov_spinbox.setRange(1, 180)
        self.fov_spinbox.setValue(60)
        self.fov_spinbox.setSuffix('°')
        self.fov_spinbox.valueChanged.connect(self._on_fov_changed)
        fov_label = BodyLabel("FOV", parent=self.gl_setting_Menu)
        fov_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        frame.layout().addWidget(fov_label)
        frame.layout().addWidget(self.fov_spinbox)
        frame.adjustSize()
        
        self.gl_setting_Menu.addWidget(frame, selectable=False)
        
        frame = QFrame()
        frame.setLayout(QHBoxLayout())
        frame.layout().setContentsMargins(0, 10, 0, 10)
        frame.layout().setSpacing(20)
        self.far_spinbox = SpinBox(parent=self.gl_setting_Menu)
        self.far_spinbox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        self.far_spinbox.setRange(1, 100000)
        self.far_spinbox.setValue(4000)
        self.far_spinbox.setSuffix('m')
        self.far_spinbox.valueChanged.connect(self._on_far_changed)
        far_label = BodyLabel("Far", parent=self.gl_setting_Menu)
        far_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.near_spinbox = DoubleSpinBox(parent=self.gl_setting_Menu)
        self.near_spinbox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        self.near_spinbox.setRange(0.001, 10)
        self.near_spinbox.setValue(0.100)
        self.near_spinbox.setSingleStep(0.001)
        self.near_spinbox.setSuffix('m')
        self.near_spinbox.valueChanged.connect(self._on_near_changed)
        near_label = BodyLabel("Near", parent=self.gl_setting_Menu)
        near_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        frame.layout().addWidget(near_label)
        frame.layout().addWidget(self.near_spinbox)
        
        frame.layout().addWidget(far_label)
        frame.layout().addWidget(self.far_spinbox)
        frame.adjustSize()
        
        self.gl_setting_Menu.addWidget(frame, selectable=False)
        
        self.gl_setting_Menu.addSeparator()
        
        frame = QFrame()
        frame.setLayout(QHBoxLayout())
        frame.layout().setContentsMargins(0, 10, 0, 10)
        frame.layout().setSpacing(20)
        grid_control_toggle = SwitchButton(parent=self.gl_setting_Menu)
        grid_control_toggle.setChecked(True)
        grid_control_toggle.checkedChanged.connect(self._on_grid_visibility_changed)
        grid_control_label = BodyLabel("Grid Visibility", parent=self.gl_setting_Menu)
        frame.layout().addWidget(grid_control_label)
        frame.layout().addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        frame.layout().addWidget(grid_control_toggle)
        frame.adjustSize()
        self.gl_setting_Menu.addWidget(frame, selectable=False)
        
        
        frame = QFrame()
        frame.setLayout(QHBoxLayout())
        frame.layout().setContentsMargins(0, 10, 0, 10)
        frame.layout().setSpacing(20)
        axis_control_toggle = SwitchButton(parent=self.gl_setting_Menu)
        axis_control_toggle.setChecked(True)
        axis_control_toggle.checkedChanged.connect(self._on_axis_visibility_changed)
        axis_control_label = BodyLabel("Axis Visibility", parent=self.gl_setting_Menu)
        frame.layout().addWidget(axis_control_label)
        frame.layout().addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        frame.layout().addWidget(axis_control_toggle)
        frame.adjustSize()
        self.gl_setting_Menu.addWidget(frame, selectable=False)
        
        
        frame = QFrame()
        frame.setLayout(QHBoxLayout())
        frame.layout().setContentsMargins(0, 10, 0, 10)
        frame.layout().setSpacing(20)
        enable_ssao_toggle = SwitchButton(parent=self.gl_setting_Menu)
        enable_ssao_toggle.setChecked(True)
        enable_ssao_toggle.checkedChanged.connect(self._on_ssao_visibility_changed)
        enable_ssao_label = BodyLabel("Enable SSAO", parent=self.gl_setting_Menu)
        frame.layout().addWidget(enable_ssao_label)
        frame.layout().addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        frame.layout().addWidget(enable_ssao_toggle)
        frame.adjustSize()
        self.gl_setting_Menu.addWidget(frame, selectable=False)

        frame = QFrame()
        frame.setLayout(QHBoxLayout())
        frame.layout().setContentsMargins(0, 10, 0, 10)
        frame.layout().setSpacing(20)
        axis_size_slider = Slider(parent=self.gl_setting_Menu)
        axis_size_slider.setOrientation(Qt.Horizontal)
        axis_size_slider.setRange(1, 100)
        axis_size_slider.setValue(1)
        axis_size_slider.valueChanged.connect(self._on_axis_length_changed)
        axis_size_label = BodyLabel("Axis Size", parent=self.gl_setting_Menu)
        frame.layout().addWidget(axis_size_label)
        frame.layout().addWidget(axis_size_slider)
        frame.adjustSize()
        self.gl_setting_Menu.addWidget(frame, selectable=False)


        self.gl_setting_Menu.addSeparator()

        action_resetCamera = Action(FIF.CANCEL, 'Reset Camera')
        action_resetCamera.triggered.connect(self._on_reset_camera)
        # action_resetCamera.setShortcut(QKeySequence("DoubleClick"))
        
        self.gl_setting_Menu.addActions([
            action_resetCamera,
        ])
        
        self.gl_setting_Menu.addSeparator()
        
        # 深度图相关功能
        
        action_saveDepth = Action(FIF.SAVE, 'Save Depth Maps')
        action_saveDepth.triggered.connect(self._on_save_depth)

        action_saveRGBA = Action(FIF.SAVE, 'Save RGBA Maps')
        action_saveRGBA.triggered.connect(self._on_save_rgba)

        self.gl_setting_Menu.addActions([
            action_saveDepth,
            action_saveRGBA
        ])
        
        self.gl_setting_Menu.addSeparator()
        
        self.action_github = Action(FIF.GITHUB, 'GitHub')
        self.action_github.triggered.connect(lambda: webbrowser.open('https://github.com/KaifengT/Batch3D'))

        self.gl_setting_Menu.addActions([
            self.action_github,
        ])

        self.gl_setting_button.setMenu(self.gl_setting_Menu)
        self.gl_setting_button.adjustSize()
    
    def _on_render_mode_changed(self, mode):
        if self.render_mode_callback:
            self.render_mode_callback(mode)
    
    def _on_camera_control_changed(self, index):
        if self.camera_control_callback:
            self.camera_control_callback(index)
    
    def _on_camera_persp_changed(self, index):
        if self.camera_persp_callback:
            self.camera_persp_callback(index)
            
    def _on_camera_view_changed(self, index):
        if self.camera_view_callback:
            self.camera_view_callback(index)
            
    def _on_fov_changed(self, value):
        if self.fov_callback:
            self.fov_callback(value)
            
    def _on_far_changed(self, value):
        if self.far_callback:
            self.far_callback(value)
            
    def _on_near_changed(self, value):
        if self.near_callback:
            self.near_callback(value)
            
    def _on_grid_visibility_changed(self, state):
        if self.grid_vis_callback:
            self.grid_vis_callback(state)

    def _on_axis_visibility_changed(self, state):
        if self.axis_vis_callback:
            self.axis_vis_callback(state)
            
    def _on_axis_length_changed(self, length):
        if self.axis_length_callback:
            self.axis_length_callback(length)
            
    def _on_reset_camera(self):
        if self.reset_camera_callback:
            self.reset_camera_callback()
    
    def _on_save_depth(self):
        if self.save_depth_callback:
            self.save_depth_callback()
    
    def _on_save_rgba(self):
        if self.save_rgba_callback:
            self.save_rgba_callback()
            
    def _on_ssao_visibility_changed(self, state):
        if self.enable_ssao_callback:
            self.enable_ssao_callback(state)


    def move(self, x, y):
        self.gl_setting_button.move(x, y)
    
    def get_button(self):
        return self.gl_setting_button
    
    def get_menu(self):
        return self.gl_setting_Menu

