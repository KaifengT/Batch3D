'''
copyright: (c) 2024 by KaifengTang, TingruiGuo
'''
import sys, os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
import math
import time
import traceback
import numpy as np
from PySide6.QtCore import (QTimer, Qt, QRect, QRectF, Signal, QSize, QObject, QPoint, QKeyCombination)
from PySide6.QtGui import (QBrush, QColor,QWheelEvent,QMouseEvent, QPainter, QPen, QFont, QKeySequence)
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QCheckBox, QSizePolicy, QVBoxLayout, QFrame, QHBoxLayout, QSpacerItem)
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.arrays import vbo
from types import NoneType
import trimesh.visual
from tools.overload import singledispatchmethod
from .utils.transformations import rotation_matrix, rpy2hRT, invHRT
from .utils.kalman import kalmanFilter
from typing import Tuple
import copy
from .mesh import Mesh, PointCloud, Grid, Axis, BoundingBox, Lines, Arrow, BaseObject
from .utils.objloader import OBJ
from .utils.transformations import quaternion_from_matrix, quaternion_matrix
from ui.addon import GLAddon_ind
from ui.statusBar import StatusBar
import trimesh
from enum import Enum
# from memory_profiler import profile

from qfluentwidgets import CheckBox, setCustomStyleSheet, ComboBox, Slider, SegmentedWidget, DropDownToolButton, \
    RoundMenu, Action, BodyLabel, SpinBox, DoubleSpinBox, ToggleButton, SwitchButton
from qfluentwidgets import FluentIcon as FIF






class GLCamera(QObject):
    
    class controlType(Enum):
        arcball = 0
        trackball = 1
        
    class projectionMode(Enum):
        perspective = 0
        orthographic = 1
        
    
    updateSignal = Signal()
    
    '''
           y
           ^
           |
         __|__
        | -z  |----> x
        |_____|
    
    '''
    def __init__(self) -> None:
        super().__init__()
            
        self.azimuth=135
        self.elevation=-55
        self.viewPortDistance = 10
        self.CameraTransformMat = np.identity(4)
        self.lookatPoint = np.array([0., 0., 0.,])
        self.fy = 1
        self.intr = np.eye(3)
        
        self.viewAngle = 60.0
        self.near = 0.1
        self.far = 4000.0
        
        self.controltype = self.controlType.arcball
        
        self.archball_rmat = None
        self.target = None
        self.archball_radius = 1.5
        self.reset_flag = False
        
        self.arcboall_quat = np.array([1, 0, 0, 0])
        self.last_arcboall_quat = np.array([1, 0, 0, 0])
        self.arcboall_t = np.array([0, 0, 0])
        
        self.filterAEV = kalmanFilter(3, R=0.2)
        self.filterlookatPoint = kalmanFilter(3, R=0.2)
        # BUG TO FIX: filterRotaion cannot deal with quaternion symmetry when R >> Q
        # self.filterRotaion = kalmanFilter(4, Q=0.5, R=0.1)
        self.filterRotaion = kalmanFilter(4, R=0.4)
        self.filterAngle = kalmanFilter(1)
        
        # 投影相关的滤波器
        self.filterPersp = kalmanFilter(16, R=0.5)  # 4x4 投影矩阵
        self.filterViewAngle = kalmanFilter(1, R=0.1)  # FOV角度
        self.filterNear = kalmanFilter(1, R=0.1)  # 近平面
        self.filterFar = kalmanFilter(1, R=0.1)  # 远平面
        
        # 投影矩阵缓存
        self.currentProjMatrix = None
        self.targetProjMatrix = None
        
        self.projection_mode = self.projectionMode.perspective
        
        self.filterAEV.stable(np.array([self.azimuth, self.elevation, self.viewPortDistance]))
        self.updateTransform(False, False)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateTransform)
        self.timer.setSingleShot(False)
        self.timer.setInterval(7)
        self.timer.start()


        self.timer_proj = QTimer()
        self.timer_proj.timeout.connect(self.updateProjTransform)
        self.timer_proj.setSingleShot(False)
        self.timer_proj.setInterval(7)
        
        self.aspect = 1.0

    def setCamera(self, azimuth=0, elevation=50, distance=10, lookatPoint=np.array([0., 0., 0.,])) -> np.ndarray:
        if self.controltype == self.controlType.trackball:
            self.azimuth=azimuth
            self.elevation=elevation
            self.viewPortDistance = distance
            self.lookatPoint = lookatPoint
            
            
        else:
            rmat = rpy2hRT(0, 0, 0, 0, 0, azimuth/180.*math.pi)
            rmat =  rpy2hRT(0, 0, 0, elevation/180.*math.pi, 0, 0) @ invHRT(rmat)
            self.arcboall_quat = quaternion_from_matrix(rmat)

            self.viewPortDistance = distance    
            self.lookatPoint = lookatPoint
            
            
        return self.updateTransform()
    
    def setCameraTransform(self, transform: np.ndarray) -> np.ndarray:
        self.arcball_quat = quaternion_from_matrix(transform)
        return self.updateTransform()

    def updateIntr(self, window_h, window_w, PixelRatio=1.):
        
        self.fy = int(window_h * PixelRatio)/ 2 / math.tan(self.viewAngle /360 * math.pi)

        self.intr = np.array(
            [[self.fy, 0,            (window_w * PixelRatio)//2],
            [0,          self.fy,    (window_h * PixelRatio)//2],
            [0,          0,            1]])
        
    def resetAE(self,):
        
        # print(self.elevation, self.azimuth)
        
        if self.elevation > 0:
            counte = self.elevation // 360
            self.elevation -= counte * 360.
        else:
            counte = self.elevation // -360
            self.elevation -= counte * -360.

        if self.azimuth > 0:
            counta = self.azimuth // 360
            self.azimuth -= counta * 360.
        else:
            counta = self.azimuth // -360
            self.azimuth -= counta * -360.
        
    def map2Sphere(self, x, y, height, width):
        cx, cy = width / 2, height / 2  # center of the window
        norm_x = (x - cx) / cx
        norm_y = (cy - y) / cy

        d = math.sqrt(norm_x**2 + norm_y**2)
        if d < self.archball_radius:
            z = math.sqrt(self.archball_radius**2 - d**2)
        else:
            norm = math.sqrt(norm_x**2 + norm_y**2 + 1)
            norm_x /= norm
            norm_y /= norm
            z = 1 / norm

        return np.array([norm_x, norm_y, z])

    def calculateRotation(self, start=[0, 0], end=[0, 0]):
        # params:
        # start: 1x2 norm array, start point of the mouse drag, [x1, y1]
        # end: 1x2 norm array, end point of the mouse drag, [x2, y2]
        axis = np.cross(start, end)
        cos_angle = np.dot(start, end) / (np.linalg.norm(start) * np.linalg.norm(end))
        angle = math.acos(max(min(cos_angle, 1), -1))

        if np.allclose(axis, [0, 0, 0]):
            return [0, 0, 0], 0
        else:
            axis = axis / np.linalg.norm(axis)
            return axis, angle

    def rotationMatrixFromAxisAngle(self, axis, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        x, y, z = axis

        return np.array([
            [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])

    def rpyFromRotationMatrix(self, R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.degrees(np.array([x, y, z]))

    def rotate(self, start=0, end=0, window_h=0, window_w=0):
        # params:
        # start: 1x2 array, start point of the mouse drag, [x1, y1]
        # end: 1x2 array, end point of the mouse drag, [x2, y2]
        # window_h: int, height of the window
        # window_w: int, width of the window
        # reutrn: 4x4 np.ndarray, rotation matrix

        # map to sphere
        if self.controltype == self.controlType.trackball:
            self.azimuth -= float(start) * 0.15
            self.elevation  += float(end) * 0.15

        else:
            start_norm = self.map2Sphere(start[0], start[1], window_h, window_w)
            end_norm = self.map2Sphere(end[0], end[1], window_h, window_w)
            axis, angle = self.calculateRotation(start_norm, end_norm)
            # print('axis, angle', axis, angle)
            # transform screen space to world space
            angle *= 16
            axis = self.CameraTransformMat[:3,:3].T.dot(axis)
            if angle == 0:
                rmat = np.eye(3)
            else:
                rmat = self.rotationMatrixFromAxisAngle(axis, angle)
            # transform 3x3 rmat to 4x4 rmat
            temp_rmat = np.zeros((4,4))
            if rmat.shape == (3, 3, 3):
                print('error rmat shape', rmat.shape)

            temp_rmat[:3,:3] = rmat
            temp_rmat[3,3] = 1
            self.archball_rmat =  temp_rmat
            
            rmat = self.CameraTransformMat[:3,:3] @ rmat.T
            
            last_quat = quaternion_from_matrix(self.CameraTransformMat)
            
            targetTransformMat = np.identity(4)
            targetTransformMat[:3,:3] = rmat
            tmat = np.identity(4)
            tmat[:3,3] = self.lookatPoint.T
            # print(tmat)
            targetTransformMat = targetTransformMat @ invHRT(tmat)
            self.arcboall_quat = quaternion_from_matrix(targetTransformMat)
            
            # angle = np.dot(last_quat, self.arcboall_quat)
            # if angle < 0:
            #     print('reverse')
            #     self.arcboall_quat = -self.arcboall_quat
            
            self.arcboall_t = targetTransformMat[:3,3]
            
    def zoom(self, ddistance=0):
        self.viewPortDistance -= ddistance * self.viewPortDistance * 0.1
        
    def translate(self, x=0, y=0,):
        self.updateTransform()
        
        scale = self.viewPortDistance * 1e-3
        
        xvec = np.array([-scale,0.,0.,0.]).T @ self.CameraTransformMat
        yvec = np.array([0.,scale,0.,0.]).T @ self.CameraTransformMat
        xdelta = xvec * x
        ydelta = yvec * y
        self.lookatPoint += xdelta[:3]
        self.lookatPoint += ydelta[:3]
        
    def translateTo(self, x=0, y=0, z=0, isAnimated=False, isEmit=True):
        self.lookatPoint = np.array([x, y, z,], dtype=np.float32)
        self.updateTransform(isAnimated=isAnimated, isEmit=isEmit)
        
    def rotationMatrix2Quaternion(self, R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return np.array([qw, qx, qy, qz])

    def quaternion2RotationMatrix(self, q):
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q

        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        return R

    def updateTransform(self, isAnimated=True, isEmit=True) -> np.ndarray:
        # TODO:
        # add timer.stop() when the screen is not moving
        
        if self.controltype == self.controlType.arcball:
    
            if isAnimated:
                
                if not self.timer.isActive():
                    self.timer.start()
                # self.timer.start()
                
                aev_in = np.array([self.azimuth, self.elevation, self.viewPortDistance])
                aev = self.filterAEV.forward(aev_in)
                lookatPoint = self.filterlookatPoint.forward(self.lookatPoint)
                

                if np.dot(self.arcboall_quat, self.last_arcboall_quat) < 0:
                    self.arcboall_quat = -self.arcboall_quat

                quat = self.filterRotaion.forward(self.arcboall_quat)
                if np.allclose(aev, aev_in, atol=1e-3) and \
                    np.allclose(lookatPoint, self.lookatPoint, atol=1e-3) and \
                        np.allclose(self.arcboall_quat, quat, atol=1e-5):
                    
                    self.timer.stop()
                                
                

                tmat = np.identity(4)
                tmat[:3,3] = lookatPoint

                self.last_arcboall_quat = copy.deepcopy(self.arcboall_quat)
                self.CameraTransformMat = np.identity(4)
                self.CameraTransformMat = quaternion_matrix(quat)
                
                self.CameraTransformMat[2, 3] = -aev[2]
                
                self.CameraTransformMat = self.CameraTransformMat @ invHRT(tmat)
                

            else:
                
                self.filterlookatPoint.stable(self.lookatPoint)
                
                rmat = rpy2hRT(0,0,0,0,0,self.azimuth/180.*math.pi)
                self.CameraTransformMat =  rpy2hRT(0,0,0,self.elevation/180.*math.pi,0,0) @ np.linalg.inv(rmat) 
                self.CameraTransformMat[2, 3] = -self.viewPortDistance
                
                tmat = np.identity(4)
                tmat[:3,3] = self.lookatPoint.T
                self.CameraTransformMat = self.CameraTransformMat @ invHRT(tmat)
                
            if isEmit:
                self.updateSignal.emit()
            return self.CameraTransformMat            
        else:
            
    
    
            if isAnimated:
                
                if not self.timer.isActive():
                    self.timer.start()
                
                aev_in = np.array([self.azimuth, self.elevation, self.viewPortDistance])
                aev = self.filterAEV.forward(aev_in)
                lookatPoint = self.filterlookatPoint.forward(self.lookatPoint)
                
                if np.allclose(aev, aev_in, atol=1e-3) and np.allclose(lookatPoint, self.lookatPoint, atol=1e-3):
                    # print('stop')
                    self.timer.stop()
                    self.resetAE()
                    self.filterAEV.stable(np.array([self.azimuth, self.elevation, self.viewPortDistance]))
                
                rmat = rpy2hRT(0, 0, 0, 0, 0, aev[0]/180.*math.pi)
                self.CameraTransformMat =  rpy2hRT(0, 0, 0, aev[1]/180.*math.pi, 0, 0) @ invHRT(rmat) 
                self.CameraTransformMat[2, 3] = -aev[2]
                
                tmat = np.identity(4)
                tmat[:3,3] = lookatPoint.T
                self.CameraTransformMat = self.CameraTransformMat @ invHRT(tmat)
                        
            else:
                
                self.filterlookatPoint.stable(self.lookatPoint)
                
                rmat = rpy2hRT(0,0,0,0,0,self.azimuth/180.*math.pi)
                self.CameraTransformMat =  rpy2hRT(0,0,0,self.elevation/180.*math.pi,0,0) @ np.linalg.inv(rmat) 
                self.CameraTransformMat[2, 3] = -self.viewPortDistance
                
                tmat = np.identity(4)
                tmat[:3,3] = self.lookatPoint.T
                self.CameraTransformMat = self.CameraTransformMat @ np.linalg.inv(tmat)
                
            if isEmit:
                self.updateSignal.emit()
            return self.CameraTransformMat
            
    def rayVector(self, ViewPortX=0, ViewPortY=0, dis=1) -> np.ndarray:
        '''
        return: np.ndarray(3, )
        '''
        
        xymap = np.linalg.inv(self.intr) @ np.array([ViewPortX, ViewPortY, 1]).T
        xymap = np.multiply(xymap, -dis)
        pvec = np.array([-xymap[0],-xymap[1],xymap[2],1]).T
        return np.linalg.inv(self.CameraTransformMat) @ pvec

    def updateProjTransform(self, isAnimated=True, isEmit=True) -> np.ndarray:
        # 计算目标投影矩阵
        if self.projection_mode == self.projectionMode.perspective:
            
            right = np.tan(np.radians(self.viewAngle/2)) * self.near
            left = -right
            top = right/self.aspect
            bottom = left/self.aspect
            rw, rh, rd = 1/(right-left), 1/(top-bottom), 1/(self.far-self.near)
    
            target_matrix = np.array([
                [2 * self.near * rw, 0, 0, 0],
                [0, 2 * self.near * rh, 0, 0],
                [(right+left) * rw, (top+bottom) * rh, -(self.far+self.near) * rd, -1],
                [0, 0, -2 * self.near * self.far * rd, 0]
            ], dtype=np.float32)
            
            
        elif self.projection_mode == self.projectionMode.orthographic:

            height = self.viewPortDistance * 0.2
            width = height * self.aspect
            right = width
            left = -width
            top = height
            bottom = -height
            
            rw, rh, rd = 1/(right-left), 1/(top-bottom), 1/(self.far-self.near)

            target_matrix = np.array([
                [2. * rw, 0, 0, 0],
                [0, 2. * rh, 0, 0],
                [0, 0, -2. * rd, -(self.far+self.near) * rd],
                [0, 0, 0, 2.]
            ], dtype=np.float32).T
            
            
        else:
            raise ValueError(f'Unknown projection mode: {self.projection_mode}')
        
        if isAnimated:
            if self.currentProjMatrix is None:
                self.currentProjMatrix = target_matrix.copy()
                self.filterPersp.stable(target_matrix.flatten())
                return target_matrix
            
            smoothed_matrix = self.filterPersp.forward(target_matrix.flatten())
            self.currentProjMatrix = smoothed_matrix.reshape(4, 4)
            
            if np.allclose(self.currentProjMatrix, target_matrix, atol=1e-4):
                if self.timer_proj.isActive():
                    self.timer_proj.stop()
                    self.filterPersp.stable(self.currentProjMatrix.flatten())
                    # print('Projection matrix animation stopped.')
                return self.currentProjMatrix.astype(np.float32)
            
            if not self.timer_proj.isActive():
                self.timer_proj.start()
                # print('Projection matrix animation started.')
                
            if isEmit:
                self.updateSignal.emit()
            
            return self.currentProjMatrix.astype(np.float32)
        else:
            self.currentProjMatrix = target_matrix.copy()
            self.filterPersp.stable(target_matrix.flatten())
            if self.timer_proj.isActive():
                self.timer_proj.stop()
            return target_matrix

    def setFOV(self, fov=60.0):
        self.viewAngle = fov
        # if not self.timer_proj.isActive():
        #     self.timer_proj.start()
        self.updateSignal.emit()
        
    def setNear(self, near=0.1):
        if near <= 0.0001:
            near = 0.0001
        self.near = near
        self.updateSignal.emit()
        
    def setFar(self, far=4000.0):
        if far >= 100000:
            far = 100000
        if far <= self.near + 0.0001:
            far = self.near + 0.0001
        self.far = far
        self.updateSignal.emit()

    def setAspectRatio(self, aspect_ratio):
        self.aspect = aspect_ratio
        # self.updateSignal.emit()

    def setViewPreset(self, preset=0):

        presets = {
            0: (90,  -90, self.viewPortDistance), # +X
            1: (-90, -90, self.viewPortDistance), # -X
            2: (180, -90, self.viewPortDistance), # +Y
            3: (0,   -90, self.viewPortDistance), # -Y
            4: (0,     0, self.viewPortDistance), # +Z
            5: (0,   180, self.viewPortDistance), # -Z
        }
        
        
        if preset in presets:
            azimuth, elevation, distance = presets[preset]
            self.setCamera(azimuth=azimuth, elevation=elevation, 
                         distance=distance, lookatPoint=self.lookatPoint)


     

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
                 axis_length_callback=None,):
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

        action_resetCamera = Action('Reset Camera')
        action_resetCamera.triggered.connect(self._on_reset_camera)
        # action_resetCamera.setShortcut(QKeySequence("DoubleClick"))
        
        self.gl_setting_Menu.addActions([
            action_resetCamera,
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
            
        self.gl_camera_perp_combobox.setCurrentItem('1')
        self._on_camera_persp_changed(1)  # Set to ortho mode when changing view

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
    
    def move(self, x, y):
        self.gl_setting_button.move(x, y)
    
    def get_button(self):
        return self.gl_setting_button
    
    def get_menu(self):
        return self.gl_setting_Menu


class GLWidget(QOpenGLWidget):


    def __init__(self, 
        parent: QWidget=None,
        background_color: Tuple = (0, 0, 0, 0),
        **kwargs,
        ) -> None:

        super().__init__(parent)

        self.parent = parent
        self.setMinimumSize(200, 200)
        self.window_w = 0
        self.window_h = 0


        # For macOS
        if sys.platform == 'darwin':
            background_color = [0.109, 0.117, 0.125, 1.0]
            self.font = QFont(['SF Pro Display', 'Helvetica Neue', 'Arial'], 10, QFont.Weight.Normal)
        
        # For Windows
        elif sys.platform == 'win32':
            background_color = [0., 0., 0., 0.]
            self.font = QFont([u'Cascadia Mono', u'Microsoft Yahei UI'], 9, )

        else:
            background_color = [0.109, 0.117, 0.125, 1.0]
            self.font = QFont([u'Cascadia Mono', u'Microsoft Yahei UI'], 9, )



        self.bg_color = background_color


        self.objectList = {}

        self.worldTextList = {}
        

        self.statusbar = StatusBar(self)
        self.statusbar.setHidden(True)



        self.lookat_point = [0,0,0]
        self.elevation_angle = 0
        
        self.lastU = 0
        self.lastV = 0
        self.textShift = 0
        self.lastPos = QPoint(0, 0)
        
        self.MouseClickPointinWorldCoordinate = np.array([0,0,0,1])

        self.canonicalModelMatrix = np.identity(4, dtype=np.float32)
        
        self.filter = kalmanFilter(7)
        
        self.scale = 1.0
        
        self.axis_scale = 1.0
                
        self.tempMat = np.identity(4, dtype=np.float32)
        
        self.camera = GLCamera()
        self.camera.updateSignal.connect(self.update)
        # self.camera.updateSignal.connect(self.updateIndicator)
        
        self.textPainter = QPainter()
        # self.flush_timer.start()
        
        self.labelSwitchList = {}
        self.labelSwitchStatue = {}
        
        #----- MSAA 4X -----#
        GLFormat = self.format()
        GLFormat.setSamples(4)
        self.setFormat(GLFormat)
        
        self.objMap = {
            'p':PointCloud,
            'l':Lines,
            'b':BoundingBox,
            'a':Arrow,
            'm':Mesh,
        }
        
        self.isAxisVisable = True
        self.isGridVisable = True
        
        self.key_light_dir = np.array([1.2, 1.5, 1.1], dtype=np.float32) * 10000
        self.key_light_color = np.array([0.3, 0.4, 0.4], dtype=np.float32)

        self.fill_light_dir = np.array([1, 0.2, 0.1], dtype=np.float32) * 10000
        self.fill_light_color = np.array([0.3, 0.4, 0.3], dtype=np.float32)

        self.back_light_dir = np.array([-0.5, -0.5, -0.2], dtype=np.float32) * 10000
        self.back_light_color = np.array([0.4, 0.4, 0.3], dtype=np.float32)

        self.top_light_dir = np.array([0.2, 0.3, 1], dtype=np.float32) * 10000
        self.top_light_color = np.array([0.2, 0.2, 0.2], dtype=np.float32)

        self.bottom_light_dir = np.array([0.4, 0.1, -1.2], dtype=np.float32) * 10000
        self.bottom_light_color = np.array([0.2, 0.2, 0.2], dtype=np.float32)        
        
        self.light_dir = np.array([1, 1, 0], dtype=np.float32)     # 光线照射方向
        self.light_color = np.array([1, 1, 1], dtype=np.float32)    # 光线颜色
        self.ambient = np.array([0.45, 0.45, 0.45], dtype=np.float32)  # 环境光颜色
        self.shiny = 50                                             # 高光系数
        self.specular = 0.1                                       # 镜面反射系数
        self.diffuse = 1.0                                         # 漫反射系数
        self.pellucid = 0.5                                         # 透光度

        self.gl_render_mode = 1
        self.point_line_size = 3
        

        self.gl_settings = GLSettingWidget(
            parent=self,
            render_mode_callback=self.setRenderMode,
            camera_control_callback=self.setCameraControl,
            camera_persp_callback=self.setCameraPerspMode,
            camera_view_callback=self.setCameraViewPreset,
            reset_camera_callback=self.resetCamera,
            fov_callback=self.camera.setFOV,
            near_callback=self.camera.setNear,
            far_callback=self.camera.setFar,
            grid_vis_callback=self.setGridVisibility,
            axis_vis_callback=self.setAxisVisibility,
            axis_length_callback=self.setAxisScale,
        )
        
        self.gl_setting_button = self.gl_settings.get_button()
        self.gl_setting_Menu = self.gl_settings.get_menu()
        self.gl_render_mode_combobox = self.gl_settings.gl_render_mode_combobox
        self.gl_camera_control_combobox = self.gl_settings.gl_camera_control_combobox
        self.gl_camera_perp_combobox = self.gl_settings.gl_camera_perp_combobox
        self.fov_spinbox = self.gl_settings.fov_spinbox

        self.setMinimumSize(200, 200)
            
            
    def setBackgroundColor(self, color: Tuple[float, float, float, float]):
        '''
        this method sets the background color of the OpenGL widget.
        no need to use this func for Windows platform
        Args:
            color (tuple): A tuple of 4 floats representing the RGBA color values, each in the range 0-1.
        Returns:
            None
        '''
        assert len(color) == 4, "Color must be a tuple of 4 floats (R, G, B, A) in range 0-1."
        if all(0 <= c <= 1. for c in color):
            self.bg_color = color
            print(f'Setting background color to: {self.bg_color}')
            self.makeCurrent()
            glClearColor(*self.bg_color)
            self.update()
        else:
            raise ValueError("Color values must be in the range 0-1.")


    def setCameraControl(self, index):
        
        self.camera.controltype = self.camera.controlType(index)
        self.resetCamera()
        
    def setCameraPerspMode(self, index):
        self.camera.projection_mode = self.camera.projectionMode(index)
        if not self.camera.timer_proj.isActive():
            self.camera.timer_proj.start()
        self.update()
        
    def setAxisVisibility(self, isVisible=True):
        self.isAxisVisable = isVisible
        self.update()
        
    def setAxisScale(self, scale=1.0):
        self.axis_scale = scale
        scaledMatrix = np.identity(4, dtype=np.float32)
        scaledMatrix[:3,:3] *= self.axis_scale
        self.axis.setTransform(scaledMatrix)
        self.update()
    
    def setGridVisibility(self, isVisible=True):
        self.isGridVisable = isVisible
        self.update()

    def triggerFlush(self):
        self.update()
    
    def resetCamera(self):
        self.camera.setCamera(azimuth=135, elevation=-55, distance=10, lookatPoint=np.array([0., 0., 0.,]))
        if hasattr(self, 'grid'):
            self.grid.setTransform(self.grid.transformList[5])
        if hasattr(self, 'smallGrid'):
            self.smallGrid.setTransform(self.smallGrid.transformList[5])

    def setCameraViewPreset(self, preset=0):
        """
        设置相机视角预设的便捷方法
        
        Parameters:
            preset (int): 预设编号 0-5
                0: 前视图 (Front)
                1: 后视图 (Back) 
                2: 左视图 (Left)
                3: 右视图 (Right)
                4: 上视图 (Top)
                5: 下视图 (Bottom)
        """
        self.camera.setViewPreset(preset)
        self.grid.setTransform(self.grid.transformList[preset])
        self.smallGrid.setTransform(self.smallGrid.transformList[preset])
            
    def setObjectProps(self, key, props:dict):
        if key in self.objectList.keys():
            self.objectList[key].updateProps(props)
            
            
        self.update()

    def setObjTransform(self, ID=1, transform=None) -> None:
        _ID = str(ID)
        if _ID in self.objectList.keys():
            if transform is not None:
                self.objectList[_ID].setTransform(transform)
            else:
                self.objectList[_ID].setTransform(np.identity(4, dtype=np.float32))
        
        self.update()

    def updateObject(self, ID=1, obj:BaseObject=None) -> None:
        _ID = str(ID)
        if obj is not None:
            self.objectList.update({_ID:obj})
            
        else:
            if _ID in self.objectList.keys():
                self.objectList.pop(_ID)
                
        self.update()

    def setRenderMode(self, mode):
        self.gl_render_mode = mode
        self.update()
        


    def initializeGL(self):

        glEnable(GL_DEPTH_TEST)
        
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        

        glRenderMode(GL_RENDER)

        glPointSize(3)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LINE_SMOOTH)
        
        
        glEnable(GL_MULTISAMPLE)
        glShadeModel(GL_SMOOTH)
        
        glClearColor(*self.bg_color)
        glEnable(GL_COLOR_MATERIAL)

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        
        self.resetCamera()
        
        self.grid = Grid()
        self.smallGrid = Grid(n=510, scale=0.1)
        self.axis = Axis()
                
        
            
        try:
            if sys.platform == 'darwin':
                print('Using OpenGL 1.2')
                vshader_src = open('./glw/vshader_src_120.glsl', encoding='utf-8').read()
                fshader_src = open('./glw/fshader_src_120.glsl', encoding='utf-8').read()
                vshader = shaders.compileShader(vshader_src, GL_VERTEX_SHADER)
                fshader = shaders.compileShader(fshader_src, GL_FRAGMENT_SHADER)
                self.program = shaders.compileProgram(vshader, fshader, validate=False)
            else:
                print('Using OpenGL 3.3')
                vshader_src = open('./glw/vshader_src_330.glsl', encoding='utf-8').read()
                fshader_src = open('./glw/fshader_src_330.glsl', encoding='utf-8').read()
                vshader = shaders.compileShader(vshader_src, GL_VERTEX_SHADER)
                fshader = shaders.compileShader(fshader_src, GL_FRAGMENT_SHADER)
                self.program = shaders.compileProgram(vshader, fshader)
             
        except:
            print('Shader compilation failed')


        # For future use
        # splat_vshader_src = open('./glw/splat_vshader.glsl', encoding='utf-8').read()
        # splat_fshader_src = open('./glw/splat_fshader.glsl', encoding='utf-8').read()
        # splat_vshader = shaders.compileShader(splat_vshader_src, GL_VERTEX_SHADER)
        # splat_fshader = shaders.compileShader(splat_fshader_src, GL_FRAGMENT_SHADER)
        # self.splat_program = shaders.compileProgram(splat_vshader, splat_fshader)

        self.shaderAttribList = ['a_Position', 'a_Color', 'a_Normal', 'a_Texcoord']
        self.shaderUniformList = ['u_ProjMatrix', 'u_ViewMatrix', 'u_ModelMatrix', 'u_CamPos', \
                                'u_LightDir', 'u_LightColor', 'u_AmbientColor', 'u_Shiny', 'u_Specular', 'u_Diffuse', 'u_Pellucid', 'u_NumLights', \
                                'u_Lights[0].position', 'u_Lights[0].color', \
                                'u_Lights[1].position', 'u_Lights[1].color', \
                                'u_Lights[2].position', 'u_Lights[2].color', \
                                'u_Lights[3].position', 'u_Lights[3].color', \
                                'u_Lights[4].position', 'u_Lights[4].color', \
                                    'u_Texture','render_mode',
                                    'u_farPlane',
                                    'u_farPlane_ratio',
        ]

        self.shaderLocMap = {}
        
        for attrib in self.shaderAttribList:
            self.shaderLocMap.update({attrib:glGetAttribLocation(self.program, attrib)})
        
        for uniform in self.shaderUniformList:
            self.shaderLocMap.update({uniform:glGetUniformLocation(self.program, uniform)})
                
        

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.program)
        
        # reset ModelMatrix
        loc = self.shaderLocMap.get('u_ModelMatrix')
        glUniformMatrix4fv(loc, 1, GL_FALSE, self.canonicalModelMatrix, None)
        
        # set Projection and View Matrices
        loc = self.shaderLocMap.get('u_ProjMatrix')
        self.camera.setAspectRatio(float(self.window_w) / float(self.window_h))
        projMatrix = self.camera.updateProjTransform(isEmit=False)
        glUniformMatrix4fv(loc, 1, GL_FALSE, projMatrix, None)

        loc = self.shaderLocMap.get('u_ViewMatrix')
        camtrans = self.camera.updateTransform(isEmit=False)
        campos = np.linalg.inv(camtrans)[:3,3]
        
        glUniform3f(self.shaderLocMap.get('u_CamPos'), *campos)
        glUniformMatrix4fv(loc, 1, GL_FALSE, camtrans.T, None)

        
        # if self.isAxisVisable:
        #     self.axis.renderinShader(locMap=self.shaderLocMap)
            
        # if self.isGridVisable:
        #     glUniform1i(self.shaderLocMap.get('u_farPlane'), 1)
        #     glUniform1f(self.shaderLocMap.get('u_farPlane_ratio'), 0.02)

        #     self.grid.renderinShader(locMap=self.shaderLocMap)
        #     glUniform1f(self.shaderLocMap.get('u_farPlane_ratio'), 0.15)
        #     self.smallGrid.renderinShader(locMap=self.shaderLocMap)
        #     glUniform1i(self.shaderLocMap.get('u_farPlane'), 0)
        
 
        
        glUniform3f(self.shaderLocMap.get('u_Lights[0].position'), *self.key_light_dir)
        glUniform3f(self.shaderLocMap.get('u_Lights[0].color'),    *self.key_light_color)
        glUniform3f(self.shaderLocMap.get('u_Lights[1].position'), *self.fill_light_dir)
        glUniform3f(self.shaderLocMap.get('u_Lights[1].color'),    *self.fill_light_color)
        glUniform3f(self.shaderLocMap.get('u_Lights[2].position'), *self.back_light_dir)
        glUniform3f(self.shaderLocMap.get('u_Lights[2].color'),    *self.back_light_color)
        glUniform3f(self.shaderLocMap.get('u_Lights[3].position'), *self.bottom_light_dir)
        glUniform3f(self.shaderLocMap.get('u_Lights[3].color'),    *self.bottom_light_color)
        glUniform3f(self.shaderLocMap.get('u_Lights[4].position'), *self.top_light_dir)
        glUniform3f(self.shaderLocMap.get('u_Lights[4].color'),    *self.top_light_color)
        

        glUniform1i(self.shaderLocMap.get('u_NumLights'), 5)


        # loc = self.shaderLocMap.get('u_LightColor')
        # glUniform3f(loc, *self.light_color)

        loc = self.shaderLocMap.get('u_AmbientColor')
        glUniform3f(loc, *self.ambient)

        loc = self.shaderLocMap.get('u_Shiny')
        glUniform1f(loc, self.shiny)

        loc = self.shaderLocMap.get('u_Specular')
        glUniform1f(loc, self.specular)

        loc = self.shaderLocMap.get('u_Diffuse')
        glUniform1f(loc, self.diffuse)

        loc = self.shaderLocMap.get('u_Pellucid')
        glUniform1f(loc, self.pellucid)

        
        
        for k, v in self.objectList.items():
            if hasattr(v, 'renderinShader'):
                v.renderinShader(ratio=10./self.camera.viewPortDistance, locMap=self.shaderLocMap, render_mode=self.gl_render_mode, size=self.point_line_size)


        glDepthMask(GL_FALSE)


        if self.isAxisVisable:
            self.axis.renderinShader(locMap=self.shaderLocMap)
            
        if self.isGridVisable:
            glUniform1i(self.shaderLocMap.get('u_farPlane'), 1)
            glUniform1f(self.shaderLocMap.get('u_farPlane_ratio'), 0.02)

            self.grid.renderinShader(locMap=self.shaderLocMap)
            glUniform1f(self.shaderLocMap.get('u_farPlane_ratio'), 0.15)
            self.smallGrid.renderinShader(locMap=self.shaderLocMap)
            glUniform1i(self.shaderLocMap.get('u_farPlane'), 0)

        glDepthMask(GL_TRUE)


        glUseProgram(0)
        glFlush()


    def reset(self, ):
        
        
        # for k, v in self.pointcloudList.items():
        #     if hasattr(v, 'reset'):
        #         v.reset()
        # self.pointcloudList = {}
        for k, v in self.objectList.items():
            if hasattr(v, 'reset'):
                v.reset()
        self.objectList = {}
        
        # self.removeSwitchLabel()
        self.update()
        

    def resizeGL(self, w: int, h: int) -> None:
        self.window_w = w
        self.window_h = h

        self.PixelRatio = self.devicePixelRatioF()

        self.camera.updateIntr(self.window_h, self.window_w, self.PixelRatio)
        
        self.statusbar.move(0, h-self.statusbar.height())
        self.statusbar.resize(w, h)

        self.gl_settings.move((self.window_w - self.gl_setting_button.width()) - 20, 15)
        
        # print(f'GLWidget resized to {w}x{h}, PixelRatio: {self.PixelRatio}')
        return super().resizeGL(w, h)


    def drawText(self, x, y, h=18, w=50, str='', txc='#EEEEEE', bgc='#2e68c5'):
        y += 5
        glDisable(GL_DEPTH_TEST)
        painter = QPainter()
        painter.begin(self)
        painter.setFont(self.font)
        

        
        painter.setPen(QColor(*txc))
        painter.drawText(x, y , str, )

        painter.end()
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        

    def worldCoordinatetoUV(self, p):
        '''
        p: np.ndarray(1, 4)
        '''
        camCoord = self.camera.updateTransform() @ p  
        projected_coordinates = self.camera.intr @ camCoord[:3]
        projected_coordinates = projected_coordinates[:2] / projected_coordinates[2]
        projected_coordinates[0] = (self.window_w * self.PixelRatio) - projected_coordinates[0]
        return int(projected_coordinates[1]//self.PixelRatio), int(projected_coordinates[0]//self.PixelRatio)


    def worldDrawText(self, p, h=18, w=0, msg='', txc='#EEEEEE', bgc='2e68c5', lenratio=8):

        v, u = self.worldCoordinatetoUV(p)
        
        # v += 6
        # u += 15

        # l = math.sqrt((u-self.lastU)**2 + (v-self.lastV)**2)
        # if l < 12:
        #     self.textShift += 20
        # else: self.textShift = 0

        # self.lastU = u
        # self.lastV = v

        # v += self.textShift

        glDisable(GL_DEPTH_TEST)
        self.textPainter.begin(self)

        # self.textPainter.setPen(QColor('#'+bgc))
        # self.textPainter.setBrush(QColor('#BB'+bgc))
        if not w:
            w = len(msg) * lenratio
            
        # print(u, v)
        # self.textPainter.drawRoundedRect(u, v, w+10, h,  1, 1)
        
        self.textPainter.setViewport(0, 0, self.window_w * self.PixelRatio, self.window_h * self.PixelRatio)
        self.textPainter.setFont(self.font)
        self.textPainter.setPen(QColor(txc))
        self.textPainter.drawText(u, v, msg, )
        
        self.textPainter.drawRect(u, v, 10, 10)

        self.textPainter.end()
        
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
   
    ## Mouse & Key Events
    def mousePressEvent(self, event:QMouseEvent):
        self.lastPos = event.pos()
        
        self.camera.updateTransform()
        
        MouseCoordinateinViewPortX = int((self.lastPos.x()) * self.PixelRatio )
        MouseCoordinateinViewPortY = int((self.window_h -  self.lastPos.y()) * self.PixelRatio)

        self.MouseClickPointinWorldCoordinate = self.camera.rayVector(MouseCoordinateinViewPortX, MouseCoordinateinViewPortY, dis=10)

        self.update()

    def mouseMoveEvent(self, event:QMouseEvent):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        

        if event.buttons() & Qt.LeftButton:
                    
            if self.camera.controltype == self.camera.controlType.arcball:
                # archball rotation
                self.camera.rotate(
                    [event.x(), event.y()],
                    [self.lastPos.x(), self.lastPos.y()],
                    self.window_h,
                    self.window_w
                )
            else:
                # Fix up rotation
                self.camera.rotate(dx, dy)

        if event.buttons() & Qt.RightButton:
                        
            self.camera.translate(dx, dy)

        self.lastPos = event.pos()

        self.triggerFlush()

    def wheelEvent(self, event:QWheelEvent):
        angle = event.angleDelta()
            
        self.camera.zoom(angle.y()/200.)

        self.triggerFlush()

    def mouseDoubleClickEvent(self, event:QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)
        self.resetCamera()

        self.triggerFlush()
        
    # def contextMenuEvent(self, event):
    #     return self.gl_setting_Menu.exec(event.globalPos())

