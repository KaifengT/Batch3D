
import math
import time
import traceback
import numpy as np
from PySide6.QtCore import (QTimer, Qt, QRect, QRectF, Signal, QSize, QObject, QPoint)
from PySide6.QtGui import (QBrush, QColor,QWheelEvent,QMouseEvent, QPainter, QPen, QFont)
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QCheckBox, QSizePolicy, QVBoxLayout)
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.arrays import vbo
import cv2
import trimesh.visual

from .utils.transformations import rotation_matrix, rpy2hRT, invHRT
from .utils.kalman import kalmanFilter
import sys, os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from typing import Tuple
import copy
from .mesh import Mesh, PointCloud, Grid, Axis, BoundingBox, Lines, Arrow, BaseObject
from .utils.objloader import OBJ
from ui.addon import GLAddon_ind
from ui.statusBar import StatusBar
import trimesh

# from memory_profiler import profile

from qfluentwidgets import CheckBox, setCustomStyleSheet, ComboBox, Slider, SegmentedWidget

_AmbientLight  = [0.8, 0.8, 0.8, 1.0]
_DiffuseLight  = [0.5, 0.5, 0.5, 1.0]
_SpecularLight = [0.4, 0.4, 0.4, 1.0]
_PositionLight = [20.0, 20.0, 20.0, 0.0]

def_color =  np.array([    
        # [217, 217, 217],
        [233, 119, 119],
        [65 , 157, 129],
        [156, 201, 226],
        [228, 177, 240],
        [252, 205, 42 ],
        [240, 90 , 126],
        [33 , 155, 157],
        [114, 191, 120],
        [199, 91 , 122],
        [129, 180, 227],
        [115, 154, 228],
        [119, 228, 200],
        [243, 231, 155],
        [248, 160, 126],
        [206, 102, 147],
        
        ]) / 255.

# rgb(217, 217, 217),  
# rgb(233, 119, 119),        
# rgb(65 , 157, 129),
# rgb(156, 201, 226),
# rgb(228, 177, 240),
# rgb(252, 205, 42 ),
# rgb(240, 90 , 126),
# rgb(33 , 155, 157),
# rgb(114, 191, 120),
# rgb(199, 91 , 122),
# rgb(129, 180, 227),
# rgb(115, 154, 228),
# rgb(119, 228, 200),
# rgb(243, 231, 155),
# rgb(248, 160, 126),
# rgb(206, 102, 147),






checkboxStyle = '''
CheckBox{{
    color: {hexcolor};
    font: 700 10pt "Cascadia Mono", "Microsoft Yahei UI";}}
    '''

class GLCamera(QObject):
    
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
        self.elevation=-60
        self.viewPortDistance = 30
        self.CameraTransformMat = np.identity(4)
        self.lookatPoint = np.array([0., 0., 0.,])
        self.fy = 1
        self.intr = np.eye(3)
        
        self.viewAngle = 60.0
        self.near = 0.1
        self.far = 4000.0
        
        
        
        self.filterAEV = kalmanFilter(3)
        self.filterlookatPoint = kalmanFilter(3)
        
        self.filterAEV.stable(np.array([self.azimuth, self.elevation, self.viewPortDistance]))
        self.updateTransform(False, False)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateTransform)
        self.timer.setSingleShot(False)
        self.timer.setInterval(7)
        self.timer.start()
        
        
    def setCamera(self, azimuth=0, elevation=45, distance=10, lookatPoint=np.array([0., 0., 0.,])) -> np.ndarray:
        
        self.azimuth=azimuth
        self.elevation=elevation
        self.viewPortDistance = distance
        self.lookatPoint = lookatPoint
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
        
    def rotate(self, dazimuth=0, delevation=0):
        self.azimuth -= float(dazimuth) * 0.15
        self.elevation  += float(delevation) * 0.15
        
        # if self.elevation > 360.:
        #     self.elevation -= 360.
            
        # if self.elevation < 0.:
        #     self.elevation += 360.
            
        # if self.azimuth > 360.:
        #     self.azimuth -= 360.
            
        # if self.azimuth < 0.:
        #     self.azimuth += 360.
        
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
        self.lookatPoint = np.array([x, y, z,])
        self.updateTransform(isAnimated=isAnimated, isEmit=isEmit)
        
    def updateTransform(self, isAnimated=True, isEmit=True) -> np.ndarray:
    
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

       
    def updateProjTransform(self, aspect=1.) -> np.ndarray:
        right = np.tan(np.radians(self.viewAngle/2)) * self.near
        left = -right
        top = right/aspect
        bottom = left/aspect
        rw, rh, rd = 1/(right-left), 1/(top-bottom), 1/(self.far-self.near)
 
        return np.array([
            [2 * self.near * rw, 0, 0, 0],
            [0, 2 * self.near * rh, 0, 0],
            [(right+left) * rw, (top+bottom) * rh, -(self.far+self.near) * rd, -1],
            [0, 0, -2 * self.near * self.far * rd, 0]
        ], dtype=np.float32)
               


class GLWidget(QOpenGLWidget):

    current_ship_pcd_invaild_signal = Signal(bool)
    # resize_signal = Signal(tuple)

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

        self.bg_color = (background_color[i] / 255 for i in range(len(background_color)))
        
        
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
        

        self.font = QFont([u'Cascadia Mono', u'Microsoft Yahei UI'], 9, )

        self.MouseClickPointinWorldCoordinate = np.array([0,0,0,1])

        self.baseTransform = np.identity(4, dtype=np.float32)
        
        self.filter = kalmanFilter(7)
        
        self.scale = 1.0
                
        self.tempMat = np.identity(4, dtype=np.float32)
        
        self.camera = GLCamera()
        self.camera.updateSignal.connect(self.update)
        # self.camera.updateSignal.connect(self.updateIndicator)
        
        self.textPainter = QPainter()
        # self.flush_timer.start()
        
        self.labelSwitchList = {}
        self.labelSwitchStatue = {}
        
        #----- MSAA 0X -----#
        GLFormat = self.format()
        GLFormat.setSamples(4)  # 4x抗锯齿
        self.setFormat(GLFormat)
        
        self.objMap = {
            'p':PointCloud,
            'l':Lines,
            'b':BoundingBox,
            'a':Arrow,
            'm':Mesh,
        }
        
        self.isAxisVisable = True
        
        # 主光源（Key Light）
        self.key_light_dir = np.array([1.2, 1.5, 1.1], dtype=np.float32) * 10000  # 光源方向
        self.key_light_color = np.array([0.3, 0.4, 0.4], dtype=np.float32)  # 光源颜色

        # 填充光源（Fill Light）
        self.fill_light_dir = np.array([1, 0.2, 0.1], dtype=np.float32) * 10000  # 光源方向
        self.fill_light_color = np.array([0.3, 0.4, 0.3], dtype=np.float32)  # 光源颜色

        # 背光源（Back Light）
        self.back_light_dir = np.array([-0.5, -0.5, -0.2], dtype=np.float32) * 10000  # 光源方向
        self.back_light_color = np.array([0.4, 0.4, 0.3], dtype=np.float32)  # 光源颜色

        # 顶光源（Top Light）
        self.top_light_dir = np.array([0.2, 0.3, 1], dtype=np.float32) * 10000  # 光源方向
        self.top_light_color = np.array([0.2, 0.2, 0.2], dtype=np.float32)  # 光源颜色

        # 底光源（Bottom Light）
        self.bottom_light_dir = np.array([0.4, 0.1, -1.2], dtype=np.float32) * 10000  # 光源方向
        self.bottom_light_color = np.array([0.2, 0.2, 0.2], dtype=np.float32)  # 光源颜色        
        
        self.light_dir = np.array([1, 1, 0], dtype=np.float32)     # 光线照射方向
        self.light_color = np.array([1, 1, 1], dtype=np.float32)    # 光线颜色
        self.ambient = np.array([0.15, 0.15, 0.15], dtype=np.float32)  # 环境光颜色
        self.shiny = 50                                             # 高光系数
        self.specular = 1.0                                         # 镜面反射系数
        self.diffuse = 0.7                                          # 漫反射系数
        self.pellucid = 0.5                                         # 透光度

        self.gl_render_mode = 1
        
        self.gl_render_mode_combobox = SegmentedWidget(parent=self,)
        self.gl_render_mode_combobox.setFixedWidth(210)
        # self.gl_render_mode_combobox.addItems(['线框', '简单', '法线', '贴图', ])
        self.gl_render_mode_combobox.addItem('0', '线框', lambda:self.changeRenderMode(0))
        self.gl_render_mode_combobox.addItem('1', '简单', lambda:self.changeRenderMode(1))
        self.gl_render_mode_combobox.addItem('2', '法线', lambda:self.changeRenderMode(2))
        self.gl_render_mode_combobox.addItem('3', '贴图', lambda:self.changeRenderMode(3))
        self.gl_render_mode_combobox.setCurrentItem('1')
        # self.gl_render_mode_combobox.currentIndexChanged.connect(self.changeRenderMode)
        
        self.point_line_size = 3
        
        # self.gl_slider = Slider(Qt.Orientation.Vertical, parent=self)
        # self.gl_slider.setFixedHeight(200)
        # self.gl_slider.setFixedWidth(20)
        # self.gl_slider.setMaximum(10)
        # self.gl_slider.setMinimum(1)
        # self.gl_slider.setValue(self.point_line_size)
        # self.gl_slider.valueChanged.connect(self.setGlobalSize)
        
        # sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.SwitchLabel_box = QWidget(self)
        
        # self.SwitchLabel_box.move(360, 15)
        
        # self.SwitchLabel_box_layout = QVBoxLayout(self.SwitchLabel_box)
        # self.SwitchLabel_box.setLayout(self.SwitchLabel_box_layout)
        # self.SwitchLabel_box.setSizePolicy(sizePolicy)
        # self.SwitchLabel_box.setStyleSheet('background-color: rgba(123,123,123, 50);')
        
        # self.indicator = GLAddon_ind(self)
        # self.indicator.move(200, 200)
        # self.indicator.setFixedSize(200, 200)
        # self.indicator.show()
        
        self.setMinimumSize(200, 200)
        
        
    # def updateIndicator(self, ):
        
    #     hrt = np.eye(4)
    #     hrt[:3,:3] = self.camera.CameraTransformMat[:3,:3]
    #     # self.indicator.set_pose(hrt)
        
    def setGlobalSize(self, size):
        self.point_line_size = size
        self.update()
        
    def chooseColor(self):
        id = len(self.objectList.keys())
        lc = len(def_color)
        id = id % lc
        return def_color[id]

    def trigger_flush(self):
        pass
        # if self.flush_timer.isActive():
            # return
        self.update()
        
        # self.flush_timer.start()
    

    def resetCamera(self):
        self.camera.setCamera(azimuth=135, elevation=-60, distance=10, lookatPoint=np.array([0., 0., 0.,]))
        
    def _decode_HexColor_to_RGB(self, hexcolor):
        if len(hexcolor) == 6:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4))
        elif len(hexcolor) == 8:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4, 6))
        else:
            return (0.9, 0.9, 0.9, 0.9)


    def _vector_to_transform_matrix(self, vector):
        R = cv2.Rodrigues(vector)
        rt = np.identity(4, dtype=np.float32)
        rt[:3,:3] = R[0]
        return rt
        
        
    # def removeSwitchLabel(self, name=None):
    #     if name is None:
    #         for k, v in self.labelSwitchList.items():
    #             v.deleteLater()
    #         self.labelSwitchList = {}
        
    #     if name in self.labelSwitchList.keys():
    #         self.SwitchLabel_box_layout.removeWidget(self.labelSwitchList[name])
    #         self.labelSwitchList[name].deleteLater()
    #         self.labelSwitchList.pop(name)
        
    #     # self.SwitchLabel_box.update()
    #     # self.SwitchLabel_box.adjustSize()
    #     self.SwitchLabel_box_layout.invalidate()
    #     self.SwitchLabel_box_layout.activate()
    #     self.SwitchLabel_box.adjustSize()
        
        
    # def addSwitchLabel(self, name, color=None):
        
    #     # print(color)
    #     def _isHexColorinName(name) -> str:
    #         if '#' in name:
    #             name = name.split('#', 2)[1]
    #             name = name.split('_', 2)[0]
    #             if len(name) == 6 or len(name) == 8:
    #                 return name
    #             else:
    #                 return '808080'
    #         else:
    #             return '808080'

        
    #     def HEXRGBA2QTHEX(hex:str):
    #         if len(hex) == 6:
    #             return '#' + hex
    #         elif len(hex) == 8:
    #             return '#' + hex[-2:] + hex[:-2]
    #         else:
    #             return '#808080'
            
    #     def RGBA2HEXRGBA(rgb):
    #         if len(rgb) == 3:
    #             return '%02x%02x%02xff' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    #         elif len(rgb) == 4:
    #             return '%02x%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), int(rgb[3]*255))
    #         else:
    #             return '808080'
            
    #     def getColor(name):
    #         return HEXRGBA2QTHEX(_isHexColorinName(name)) if color is None else HEXRGBA2QTHEX(RGBA2HEXRGBA(color))
            
    #     # print(HEXRGBA2QTHEX(color))
        
    #     labelObject = CheckBox(parent=self.SwitchLabel_box, text=name,)
    #     self.SwitchLabel_box_layout.addWidget(labelObject)
    #     labelObject.setChecked(True)
        
    #     _qss = checkboxStyle.format(hexcolor=getColor(name))
    #     setCustomStyleSheet(labelObject, _qss, _qss)
        
        
    #     if name in self.labelSwitchList.keys():
    #         self.labelSwitchList[name].deleteLater()
    #         self.labelSwitchList.pop(name)
            
    #     self.labelSwitchList.update({name:labelObject})

    #     # labelObject.setStyleSheet(checkboxStyle.format(hexcolor=getColor(name)))
    #     labelObject.setFont(self.font)
    #     labelObject.show()
    #     # labelObject.move(360+15, 15 + 30 * (len(self.labelSwitchList)-1))
    #     labelObject.stateChanged.connect(lambda: self.showORHideObject(name, self.labelSwitchList[name].isChecked()))
        
    #     if name in self.labelSwitchStatue.keys():
    #         labelObject.setChecked(self.labelSwitchStatue[name])
            
            
    #     self.update()
    #     self.SwitchLabel_box_layout.invalidate()
    #     self.SwitchLabel_box_layout.activate()

    #     self.SwitchLabel_box.adjustSize()
        
    def showORHideObject(self, name, isShow=True):
        if name in self.objectList.keys():
            
            self.labelSwitchStatue.update({name:isShow})
            
            self.objectList[name].show(isShow)
            # if 'arrow'+name in self.objectList.keys():
            #     self.objectList['arrow'+name].isShow = isShow
                
            self.update()
            

    def updateObjectProps(self, key, props:dict):
        if key in self.objectList.keys():
            self.objectList[key].updateProps(props)
            
            
        self.update()

    # ``````````````
    def updateObject_API(self, ID=1, objType=None, **kwargs) -> None:
        _ID = str(ID)
        availObjTypes = self.objMap.keys()
        if objType in availObjTypes:
            # if isinstance(ID, str):
            #     cid = 0
            # else:
            #     cid = ID % len(def_color)
                
            # if color is not None:
            #     if isinstance(color, str):
            #         _color = self._decode_HexColor_to_RGB(color)
                
            # else:
            #     _color = def_color[cid]
            
            assert objType in availObjTypes, f'type must in {availObjTypes}'
            self.objectList.update({_ID:self.objMap[objType](**kwargs)})
            
            # self.addSwitchLabel(_ID)
        elif objType == 'trimesh':
            self.updateTrimeshObject(ID, **kwargs)
            
        elif objType == 'clean':
            keys = list(self.objectList.keys())
            for id in keys:
                self.objectList.pop(id)
                # self.removeSwitchLabel(id)
        else:
            if _ID in self.objectList.keys():
                
                if 'transform' in kwargs.keys():
                    self.objectList[_ID].setTransform(kwargs['transform'])
                else:
                    self.objectList.pop(_ID)
                    # self.removeSwitchLabel(_ID)
                
        self.update()


    def updateObject(self, ID=1, obj:BaseObject=None, labelColor=None) -> None:
        _ID = str(ID)
        if obj is not None:
            self.objectList.update({_ID:obj})
            
            # self.addSwitchLabel(_ID, labelColor)
        else:
            if _ID in self.objectList.keys():
                self.objectList.pop(_ID)
                
                # self.removeSwitchLabel(_ID)
                
        self.update()


    def updateTrimeshObject(self, ID=1, obj=None) -> None:
        
        
        def updateTrimesh(ID, obj:trimesh.Trimesh=None) -> None:
            # mo = Mesh(obj.vertices.view(np.ndarray).astype(np.float32),
            #           obj.faces.view(np.ndarray).astype(np.int32),
            #           norm=obj.vertex_normals.view(np.ndarray).astype(np.float32),
            #           color=obj.visual.vertex_colors /255. if isinstance(obj.visual, trimesh.visual.color.ColorVisuals) else None,
            #           texture=obj.visual.material.image if isinstance(obj.visual, trimesh.visual.texture.TextureVisuals) else None,
            #           texcoord=obj.visual.uv.view(np.ndarray).astype(np.float32) if isinstance(obj.visual, trimesh.visual.texture.TextureVisuals) and hasattr(obj.visual, 'uv') and hasattr(obj.visual.uv, 'view') else None,
            #           )
            if hasattr(obj.visual, 'material'):
                if isinstance(obj.visual.material, trimesh.visual.material.SimpleMaterial):
                    tex = obj.visual.material.image
                elif isinstance(obj.visual.material, trimesh.visual.material.PBRMaterial):
                    tex = obj.visual.material.baseColorTexture
                else:
                    tex = None
            else:
                tex = None
            
            mo = Mesh(obj.vertices.view(np.ndarray).astype(np.float32),
                      obj.faces.view(np.ndarray).astype(np.int32),
                      norm=obj.face_normals.view(np.ndarray).astype(np.float32),
                      color=obj.visual.vertex_colors /255. if isinstance(obj.visual, trimesh.visual.color.ColorVisuals) else None,
                      texture=tex,
                      texcoord=obj.visual.uv.view(np.ndarray).astype(np.float32) if isinstance(obj.visual, trimesh.visual.texture.TextureVisuals) and hasattr(obj.visual, 'uv') and hasattr(obj.visual.uv, 'view') else None,
                      faceNorm=True
                      )
            
            self.objectList.update({_ID:mo})
            
            # self.addSwitchLabel(_ID)

        
        _ID = str(ID)
        if obj is not None:
            
            if isinstance(obj, trimesh.Scene):
                meshlist = []
                for k, mesh in obj.geometry.items():
                    # print(mesh.vertices.shape, mesh.faces.shape, mesh.face_normals.shape, mesh.vertex_normals.shape)
                    # print(k)
                    meshlist.append(mesh)
                    
                mesh = trimesh.util.concatenate(meshlist)
                    
                updateTrimesh(ID, mesh)
                    
            elif isinstance(obj, trimesh.Trimesh):
                
                updateTrimesh(ID, obj)
                
            elif isinstance(obj, trimesh.PointCloud):
                mo = PointCloud(obj.vertices, obj.colors / 255.)
                self.objectList.update({_ID:mo})
                
                # self.addSwitchLabel(_ID)
                
                    
        else:
            if _ID in self.objectList.keys():
                self.objectList.pop(_ID)
                
                # self.removeSwitchLabel(_ID)
                
        self.update()
        
        
    def changeRenderMode(self, mode):
        self.gl_render_mode = mode
        self.update()
        
        

    def updateframe(self, ID=1, vertex:np.ndarray=None, ccolor=None):
        _ID = 'frame' + str(ID)
        if vertex is not None:
            
            self.objectList.update({_ID:OBJ('./axis.obj', transform=vertex), })
            
        else:
            if _ID in self.objectList.keys():
                self.objectList.pop(_ID)


    def initializeGL(self):

        glEnable(GL_DEPTH_TEST)
        
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        # glLightfv(GL_LIGHT0, GL_AMBIENT, _AmbientLight)
        # glLightfv(GL_LIGHT0, GL_DIFFUSE, _DiffuseLight)
        # glLightfv(GL_LIGHT0, GL_SPECULAR, _SpecularLight)
        # glLightfv(GL_LIGHT0, GL_POSITION, _PositionLight)

        # glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0)
        glRenderMode(GL_RENDER)

        # glEnable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        # glEnable(GL_LIGHT1)
        glPointSize(3)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LINE_SMOOTH)
        
        # glEnable(GL_SAMPLER_2D_SHADOW)
        # glEnable(GL_POINT_SMOOTH)
        # glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        glEnable(GL_MULTISAMPLE)
        glShadeModel(GL_SMOOTH)
        

        # glEnable(GL_POINT_SMOOTH)
        glClearColor(*self.bg_color)
        glEnable(GL_COLOR_MATERIAL)
        # glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        
        self.resetCamera()
        
        self.grid = Grid()
        # self.axis = OBJ('./axis.obj')
        self.axis = Axis()
        
        vshader_src = open('./glw/vshader_src2.glsl', encoding='utf-8').read()
        fshader_src = open('./glw/fshader_src2.glsl', encoding='utf-8').read()
        
        vshader = shaders.compileShader(vshader_src, GL_VERTEX_SHADER)
        fshader = shaders.compileShader(fshader_src, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vshader, fshader)


        splat_vshader_src = open('./glw/splat_vshader.glsl', encoding='utf-8').read()
        splat_fshader_src = open('./glw/splat_fshader.glsl', encoding='utf-8').read()
        
        splat_vshader = shaders.compileShader(splat_vshader_src, GL_VERTEX_SHADER)
        splat_fshader = shaders.compileShader(splat_fshader_src, GL_FRAGMENT_SHADER)
        self.splat_program = shaders.compileProgram(splat_vshader, splat_fshader)

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
        ]

        self.shaderLocMap = {}
        
        for attrib in self.shaderAttribList:
            self.shaderLocMap.update({attrib:glGetAttribLocation(self.program, attrib)})
        
        for uniform in self.shaderUniformList:
            self.shaderLocMap.update({uniform:glGetUniformLocation(self.program, uniform)})
                
        

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        
        glUseProgram(self.program)
        
        
        loc = self.shaderLocMap.get('u_ProjMatrix')
        glUniformMatrix4fv(loc, 1, GL_FALSE, self.camera.updateProjTransform(float(self.window_w) / float(self.window_h)), None)

        loc = self.shaderLocMap.get('u_ViewMatrix')
        camtrans = self.camera.updateTransform(isEmit=False)
        campos = np.linalg.inv(camtrans)[:3,3]
        glUniform3f(self.shaderLocMap.get('u_CamPos'), *self.camera.lookatPoint[:3])
        glUniformMatrix4fv(loc, 1, GL_FALSE, camtrans.T, None)

        loc = self.shaderLocMap.get('u_ModelMatrix')
        modelMatrix = np.identity(4, dtype=np.float32)
        scaledMatrix = np.identity(4, dtype=np.float32)
        scaledMatrix[:3,:3] *= self.scale
        
        glUniformMatrix4fv(loc, 1, GL_FALSE, modelMatrix, None)
        
        if self.isAxisVisable:
            self.axis.renderinShader(locMap=self.shaderLocMap)
            
        # print(campos)
        glUniformMatrix4fv(loc, 1, GL_FALSE, scaledMatrix, None)
        if self.isAxisVisable:
            glUniform1i(self.shaderLocMap.get('u_farPlane'), 1)
            self.grid.renderinShader(locMap=self.shaderLocMap)
            glUniform1i(self.shaderLocMap.get('u_farPlane'), 0)
        
 
        # loc = self.shaderLocMap.get('u_LightDir')
        # glUniform3f(loc, *self.light_dir)
        
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

        # loc = self.shaderLocMap.get('u_Shiny')
        # glUniform1f(loc, self.shiny)

        # loc = self.shaderLocMap.get('u_Specular')
        # glUniform1f(loc, self.specular)

        # loc = self.shaderLocMap.get('u_Diffuse')
        # glUniform1f(loc, self.diffuse)

        # loc = self.shaderLocMap.get('u_Pellucid')
        # glUniform1f(loc, self.pellucid)

        
        
        
        
        
        for k, v in self.objectList.items():
            if hasattr(v, 'renderinShader'):
                
                glUniformMatrix4fv(self.shaderLocMap.get('u_ModelMatrix'), 1, GL_FALSE, v.transform.T, None)
                
                v.renderinShader(ratio=10./self.camera.viewPortDistance, locMap=self.shaderLocMap, render_mode=self.gl_render_mode, size=self.point_line_size)



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
        # glViewport(0,0,int(self.window_w * self.PixelRatio),int(self.window_h * self.PixelRatio))

        self.camera.updateIntr(self.window_h, self.window_w, self.PixelRatio)
        
        self.statusbar.move(0, h-self.statusbar.height())
        self.statusbar.resize(w, h)

        self.gl_render_mode_combobox.move((self.window_w - self.gl_render_mode_combobox.width())//2 , 15)
        
        # self.indicator.move(QPoint(self.window_w - self.indicator.width()-20,self.window_h - self.indicator.height() - 20))
        # self.gl_slider.move(self.window_w - self.gl_slider.width() - 15, 75)

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
                        
            self.camera.rotate(dx, dy)

        if event.buttons() & Qt.RightButton:
                        
            self.camera.translate(dx, dy)

        self.lastPos = event.pos()

        self.trigger_flush()

    def wheelEvent(self, event:QWheelEvent):
        angle = event.angleDelta()
            
        self.camera.zoom(angle.y()/200.)

        self.trigger_flush()

    def mouseDoubleClickEvent(self, event:QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)
        self.resetCamera()

        self.trigger_flush()
        


