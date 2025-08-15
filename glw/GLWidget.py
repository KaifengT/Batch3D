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
from PySide6.QtCore import (Qt, Signal, QPoint)
from PySide6.QtGui import (QColor,QWheelEvent,QMouseEvent, QPainter, QSurfaceFormat, QFont)
from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog)
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.arrays import vbo
from OpenGL.GL.framebufferobjects import *

from .utils.kalman import kalmanFilter
from typing import Tuple
import copy
from .GLMesh import Mesh, PointCloud, Grid, Axis, BoundingBox, Lines, Arrow, BaseObject, FullScreenQuad

from ui.statusBar import StatusBar
import trimesh
from enum import Enum
from PIL import Image
# from memory_profiler import profile
from .GLCamera import GLCamera
from .GLMenu import GLSettingWidget

class FBOManager_singleton:
    '''
    FBOManager is a singleton class that manages the Frame Buffer Object (FBO) and its associated textures.
    '''
    _instance = None
    _fbo = None
    _depth_texture = None
    _color_texture = None
    _width = 0
    _height = 0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FBOManager, cls).__new__(cls)
        return cls._instance
    
    def get_fbo(self, width, height):
        '''
        Get or create a Frame Buffer Object (FBO) with a depth texture.
        If the FBO already exists and the dimensions match, it will return the existing FBO.
        Args:
            width (int): The width of the FBO.
            height (int): The height of the FBO.
        Returns:
            tuple (int, int): A tuple containing the FBO and the depth texture.
        '''

        if (self._fbo is None or 
            self._width != width or 
            self._height != height):
            # print(f"Creating FBO: {width}x{height}")
            self._create_fbo(width, height)
        # self._create_fbo(width, height)
        return self._fbo, self._depth_texture
    
    def _create_fbo(self, width, height):
        
        
        if self._fbo is not None:
            self.cleanup()
            ...
        

        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # depth attachment, necessary.
        self._depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32,
                    width, height, 0,
                    GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                GL_TEXTURE_2D, self._depth_texture, 0)
        
        # color attachment, optional.
        # self._color_texture = glGenTextures(1)
        # glBindTexture(GL_TEXTURE_2D, self._color_texture)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        #             width, height, 0,
        #             GL_RGBA, GL_UNSIGNED_BYTE, None)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        #                     GL_TEXTURE_2D, self._color_texture, 0)     
                
        # check if the framebuffer is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('FBO creation failed')
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._width = width
        self._height = height
            
            

    
    def cleanup(self):
        '''
        cleanup the FBO and its associated textures.
        '''
        try:
            if self._depth_texture is not None:
                glDeleteTextures([self._depth_texture])
                self._depth_texture = None
        except Exception as e:
            print(f"error occurred while cleaning depth texture resources: {e}")

        try:
            if self._color_texture is not None:
                glDeleteTextures([self._color_texture])
                self._color_texture = None
        except Exception as e:
            print(f"error occurred while cleaning color texture resources: {e}")

        try:
            if self._fbo is not None:
                glDeleteFramebuffers([self._fbo])
                self._fbo = None
        except Exception as e:
            print(f"error occurred while cleaning FBO resources: {e}")


class FBOManager:
    '''
    FBOManager is a class that manages the Frame Buffer Object (FBO) and its associated textures.
    '''

    def __init__(self):
        self._instance = None
        self._fbo = None
        self._depth_texture = None
        self._color_texture = None
        self._geometry_texture = None
        self._width = 0
        self._height = 0
        
        self._attachments_id = []
    
    @staticmethod
    def getFormat(internalType):
        if internalType == GL_RGBA32F:
            return GL_RGBA, GL_FLOAT
        elif internalType == GL_RGB32F:
            return GL_RGB, GL_FLOAT
        elif internalType == GL_R32F:
            return GL_RED, GL_FLOAT
        elif internalType == GL_RGB: # somthing strange may be GL_RGB8I
            return GL_RGB, GL_UNSIGNED_BYTE
        else:
            raise ValueError(f"Unsupported internal type: {internalType}")
    
    def getFBO(self, width, height, depth=False, colors=[]):
        '''
        Get or create a Frame Buffer Object (FBO) with a depth texture.
        If the FBO already exists and the dimensions match, it will return the existing FBO.
        Args:
            width (int): The width of the FBO.
            height (int): The height of the FBO.
        Returns:
            tuple (int, int): A tuple containing the FBO and the depth texture.
        '''

        if (self._fbo is None or 
            self._width != width or 
            self._height != height):
            # print(f"Creating FBO: {width}x{height}")
            self._createFBO(width, height, depth, colors)
        # self._create_fbo(width, height)
        return self._fbo, self._depth_texture
    
    def _addDepthAttachment(self, width, height):
        '''
        Add a depth attachment to the FBO.
        Args:
            width (int): The width of the FBO.
            height (int): The height of the FBO.
        Returns:
            None
        '''
        depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32,
                    width, height, 0,
                    GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                GL_TEXTURE_2D, depth_texture, 0)
        self._attachments_id.append(depth_texture)

    def _addAttachment(self, width, height, internalType, attachment=GL_COLOR_ATTACHMENT0, filter=GL_NEAREST):
        '''
        Add a color attachment to the FBO.
        Args:
            internalType (int): The internal format of the texture.
                supported types: GL_RGBA32F, GL_RGB32F, GL_R32F, GL_RGB
            attachment (int): The attachment point of the texture. Choose from GL_COLOR_ATTACHMENT0 - GL_COLOR_ATTACHMENT31.
            filter (int): The texture filter mode.
                supported types: GL_LINEAR, GL_NEAREST
        
        Returns:
            None
        '''

        format, dataType = self.getFormat(internalType)

        texID = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, texID)
        glTexImage2D(GL_TEXTURE_2D, 0, internalType,
                    width, height, 0,
                    format, dataType, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment,
                            GL_TEXTURE_2D, texID, 0)
        
        self._attachments_id.append(texID)

    def _createFBO(self, width, height, depth=False, colors=[]):

        # print(f"FBOManager: Creating FBO: {width}x{height}")
        
        if self._fbo is not None:
            # print(f"FBOManager: Cleaning up existing FBO and attachments: {self._fbo}")
            self.cleanUp()
        

        self._fbo = glGenFramebuffers(1)
        # print(f"FBOManager: Generated FBO ID: {self._fbo}")
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # depth attachment, necessary.
        if depth:
            self._addDepthAttachment(width, height)
            
        for i, iType in enumerate(colors):
            self._addAttachment(width, height, iType, attachment=GL_COLOR_ATTACHMENT0 + i, filter=GL_LINEAR)

        glDrawBuffers(len(colors), [GL_COLOR_ATTACHMENT0 + i for i in range(len(colors))])

                
        # check if the framebuffer is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print(f'FBOManager: FBO creation failed: {glCheckFramebufferStatus(GL_FRAMEBUFFER)}')
            exit(0)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._width = width
        self._height = height
        
    def bindForWriting(self):
        '''
        Bind the FBO for writing.
        This method binds the FBO for rendering, allowing subsequent OpenGL calls to render to the FBO.
        Args:
            None
        Returns:
            None
        '''
        if self._fbo is None:
            raise RuntimeError('FBOManager: FBO is not created yet. Call getFBO() first.')
        
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        
    def bindForReading(self, attachment=GL_COLOR_ATTACHMENT0):
        raise NotImplementedError("FBOManager: bindForReading() is not implemented yet.")
        
        
    def bindTextureForReading(self, textureUnit, attachmentIndex):
        '''
        Bind the texture for reading.
        This method binds the texture associated with the FBO for reading, allowing subsequent OpenGL calls to read from the texture.
        NOTE: if depth=True, the depth texture will be at index 0.
        Args:
            textureUnit (int): The texture unit to bind the texture to. Choose from GL_TEXTURE0 - GL_TEXTURE31.
            attachmentIndex (int): The index of the attachment to bind.
        Returns:
            None
        '''
        if self._fbo is None:
            raise RuntimeError('FBOManager: FBO is not created yet. Call getFBO() first.')
        
        if attachmentIndex >= len(self._attachments_id):
            raise ValueError(f'FBOManager: Invalid attachment index: {attachmentIndex}, max: {len(self._attachments_id)}.')
        
        glActiveTexture(textureUnit)
        glBindTexture(GL_TEXTURE_2D, self._attachments_id[attachmentIndex])
        
            
    def cleanUp(self):
        '''
        cleanup the FBO and its associated textures.
        '''
        # print(f'FBOManager: trying to cleanup FBO resources {self._fbo}, textures {self._attachments_id}')
        try:
            if len(self._attachments_id):
                glDeleteTextures(self._attachments_id)
                self._attachments_id = []
        except Exception as e:
            print(f"FBOManager: error occurred while cleaning texture resources: {e}")

        try:
            if self._fbo is not None:
                glDeleteFramebuffers([self._fbo])
                self._fbo = None
        except Exception as e:
            print(f"FBOManager: error occurred while cleaning FBO resources: {e}")


    def __del__(self):
        self.cleanUp()
        return super().__del__()


class DepthReader:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fbo_manager = FBOManager()
        
    def resize(self, width, height):
        self.width = width
        self.height = height
    
    def before_render(self):
        '''
        call this method before rendering to set up the FBO for depth reading.
        It will create a framebuffer object (FBO) and bind it for rendering.
        Args:
            None
        Returns:
            fbo (int): The framebuffer object ID that is bound for rendering.
        
        '''
        fbo, depth_texture = self.fbo_manager.getFBO(self.width, self.height)
        # print(f"FBO: {fbo}, Depth Texture: {depth_texture}")
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        return fbo
            

    def read_depth(self):        
        '''
        After rendering, call this method to read the depth data from the FBO.
        It will read the depth data from the currently bound FBO and return it as a numpy array.
        - Note: Make sure to call before_render() before this method.
        - Note: The depth data is in NDC (Normalized Device Coordinates) format,
        which ranges from 0.0 to 1.0, where 0.0 is the near plane and 1.0 is the far plane.
        
        Args:
            None
        Returns:
            depth_array (np.ndarray): A 2D numpy array of shape (height, width) containing the depth values.
        '''

        depth_data = glReadPixels(0, 0, self.width, self.height,
                                GL_DEPTH_COMPONENT, GL_FLOAT)
        
        depth_array = np.frombuffer(depth_data, dtype=np.float32)
        depth_array = depth_array.reshape((self.height, self.width))[::-1, :]
        
        return depth_array
        
    @staticmethod
    def convertNDC2Liner(ndc_depth:np.ndarray, camera:GLCamera):
        """
        convert NDC depth to linear depth
        - Note: This method is a little bit slow, so use it carefully.
        Args:
            ndc_depth(np.ndarray): NDC depth (0.0 to 1.0)
            camera(GLCamera): GLCamera object containing camera parameters
            
        Returns:
            linear_depth: linear depth in world coordinates
        """

        if camera.projection_mode == camera.projectionMode.perspective:
            ndc_depth = ndc_depth * 2.0 - 1.0 
            linear_depth = (2.0 * camera.near * camera.far) / (
                camera.far + camera.near - ndc_depth * (camera.far - camera.near)
            )
        else:
            linear_depth = ndc_depth * (camera.far - camera.near) + camera.near
        
        return linear_depth
    


class GLWidget(QOpenGLWidget):

    leftMouseClickSignal = Signal(np.ndarray, np.ndarray)
    rightMouseClickSignal = Signal(np.ndarray, np.ndarray)
    middleMouseClickSignal = Signal(np.ndarray, np.ndarray)
    mouseReleaseSignal = Signal(np.ndarray, np.ndarray)
    mouseMoveSignal = Signal(np.ndarray, np.ndarray)

    def __init__(self, 
        parent: QWidget=None,
        background_color: Tuple = (0, 0, 0, 0),
        **kwargs,
        ) -> None:

        super().__init__(parent)

        self.parent = parent
        self.setMinimumSize(200, 200)
        self.scaled_window_w = 0
        self.scaled_window_h = 0
        self.raw_window_w = 0
        self.raw_window_h = 0


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
        
        self.mouseClickPointinWorldCoordinate = np.array([0,0,0,1])
        self.mouseClickPointinUV = np.array([0, 0])

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
        GLFormat.setSamples(1)
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
        
        # self.key_light_pos = np.array([1.2, 1.5, 1.1], dtype=np.float32) * 10000
        # self.key_light_color = np.array([0.3, 0.3, 0.3], dtype=np.float32)

        # self.fill_light_pos = np.array([1, -1.2, 0.3], dtype=np.float32) * 10000
        # self.fill_light_color = np.array([0.2, 0.3, 0.3], dtype=np.float32)

        # self.back_light_pos = np.array([-0.5, -0.5, -0.2], dtype=np.float32) * 10000
        # self.back_light_color = np.array([0.4, 0.4, 0.3], dtype=np.float32)
        
        
        self.key_light_pos = np.array([0.0, 1.1, 1.1], dtype=np.float32) * 10000
        self.key_light_color = np.array([0.3, 0.3, 0.3], dtype=np.float32)

        self.fill_light_pos = np.array([-1.0, -1.2, -1.2], dtype=np.float32) * 10000
        self.fill_light_color = np.array([0.2, 0.3, 0.3], dtype=np.float32)

        self.back_light_pos = np.array([1.0, -0.9, 1.3], dtype=np.float32) * 10000
        self.back_light_color = np.array([0.4, 0.4, 0.3], dtype=np.float32)

        # self.top_light_pos = np.array([0.2, 0.3, 1], dtype=np.float32) * 10000
        # self.top_light_color = np.array([0.2, 0.7, 0.2], dtype=np.float32)

        # self.bottom_light_pos = np.array([0.4, 0.1, -1.2], dtype=np.float32) * 10000
        # self.bottom_light_color = np.array([0.7, 0.2, 0.2], dtype=np.float32)


        self.ambient = np.array([0.7, 0.7, 0.7], dtype=np.float32)  # 环境光颜色
        self.shiny = 50                                             # 高光系数
        self.specular = 0.1                                       # 镜面反射系数
        self.diffuse = 1.0                                         # 漫反射系数
        self.pellucid = 0.5                                         # 透光度

        self.gl_render_mode = 1
        self.point_line_size = 3
        self.enableSSAO = 1
        
        self.grid = Grid()
        self.smallGrid = Grid(n=510, scale=0.1)
        self.axis = Axis()

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
            save_depth_callback=self.saveDepthMap,
            save_rgba_callback=self.saveRGBAMap,
            enable_ssao_callback=self.setEnableSSAO,
        )
        
        self.gl_setting_button = self.gl_settings.get_button()
        self.gl_setting_Menu = self.gl_settings.get_menu()
        self.gl_render_mode_combobox = self.gl_settings.gl_render_mode_combobox
        self.gl_camera_control_combobox = self.gl_settings.gl_camera_control_combobox
        self.gl_camera_perp_combobox = self.gl_settings.gl_camera_perp_combobox
        self.gl_camera_view_combobox = self.gl_settings.gl_camera_view_combobox
        self.gl_enable_ssao_toggle = self.gl_settings.enable_ssao_toggle
        self.fov_spinbox = self.gl_settings.fov_spinbox
        

        self.setMinimumSize(200, 200)
        
        self.depthMap = None
            
            
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
        self.camera.setProjectionMode(self.camera.projectionMode(index))
        self.camera.updateIntr(self.raw_window_h, self.raw_window_w)
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
        self.camera.updateIntr(self.raw_window_h, self.raw_window_w)
        self.camera.setLockRotate(False)
        self.gl_camera_view_combobox.setCurrentItem('6')
        
        if hasattr(self, 'grid'):
            self.grid.setTransform(self.grid.transformList[5])
        if hasattr(self, 'smallGrid'):
            self.smallGrid.setTransform(self.smallGrid.transformList[5])

    def setCameraViewPreset(self, preset=0):
        """
        Setting the camera view preset.
        
        Args:
            preset (int): index from 0-6
                0: Front View
                1: Back View
                2: Left View
                3: Right View
                4: Top View
                5: Bottom View
                6: Free View
        """
        if preset > 5:
            self.resetCamera()
            self.camera.setProjectionMode(GLCamera.projectionMode.perspective)
            self.gl_camera_perp_combobox.setCurrentItem('0')
            self.camera.setLockRotate(False)
            self.camera.updateIntr(self.raw_window_h, self.raw_window_w)
        else:
            self.camera.setViewPreset(preset)
            self.grid.setTransform(self.grid.transformList[preset])
            self.smallGrid.setTransform(self.smallGrid.transformList[preset])
            self.camera.setProjectionMode(GLCamera.projectionMode.orthographic)
            self.gl_camera_perp_combobox.setCurrentItem('1')
            self.camera.setLockRotate(True)
            self.camera.updateIntr(self.raw_window_h, self.raw_window_w)
            
    def setObjectProps(self, ID, props:dict):
        '''
        Setting the properties of an object in the objectList.
        Args:
            ID (int): The ID of the object in the objectList.
            props (dict): A dictionary containing the properties to be updated.
                Available properties include:
                - 'size': Size of the object (float).
                - 'isShow': Visibility of the object (boolean).
        Returns:
            None
        '''
        
        _ID = str(ID)
        if _ID in self.objectList.keys():
            self.objectList[_ID].updateProps(props)

        self.update()

    def setObjTransform(self, ID=1, transform=None) -> None:
        '''
        Setting the transformation matrix of an object in the objectList.
        Args:
            ID (int): The ID of the object in the objectList.
            transform (np.ndarray(4, 4)): The homogeneous transformation matrix to be set.
                If None, the transformation matrix will be set to the identity matrix.
        Returns:
            None
        '''
        _ID = str(ID)
        if _ID in self.objectList.keys():
            if transform is not None:
                self.objectList[_ID].setTransform(transform)
            else:
                self.objectList[_ID].setTransform(np.identity(4, dtype=np.float32))
        
        self.update()

    def updateObject(self, ID=1, obj:BaseObject=None) -> None:
        '''
        Update the object in the objectList with a new object or remove it if obj is None.
        Args:
            ID (int): The ID of the object in the objectList.
            obj (BaseObject): The new object to be set. 
                If None, the object which name matches the ID will be removed from the list.
        Returns:
            None
        '''
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

    @staticmethod
    def buildShader(vshader_path, fshader_path):
        '''
        Compile and link the vertex and fragment shaders.
        Args:
            vshader_src (str): The source code PATH of the vertex shader.
            fshader_src (str): The source code PATH of the fragment shader.
        Returns:
            program (int): The OpenGL program ID.
        '''
        try:
            vshader_src = open(vshader_path, encoding='utf-8').read()
            fshader_src = open(fshader_path, encoding='utf-8').read()

            vshader = shaders.compileShader(vshader_src, GL_VERTEX_SHADER)
            fshader = shaders.compileShader(fshader_src, GL_FRAGMENT_SHADER)
            program = shaders.compileProgram(vshader, fshader)
            return program
        except Exception as e:
            print(f"Error compiling/linking shaders: {e}")
            traceback.print_exc()
            return None
        
    @staticmethod
    def cacheShaderLocMap(program, attribList, uniformList):
        shaderLocMap = {}
        for attrib in attribList:
            shaderLocMap.update({attrib:glGetAttribLocation(program, attrib)})

        for uniform in uniformList:
            shaderLocMap.update({uniform:glGetUniformLocation(program, uniform)})
        return shaderLocMap

    def generateSSAOKernel(self, kernel_size=64):
        """
        Generate SSAO kernel samples.
        Args:
            kernel_size (int): Number of kernel samples, default is 64.
        Returns:
            kernel (np.ndarray (kernel_size, 3)): Array of sample vectors with shape (kernel_size, 3).
        """
        # Generate random vectors in the range [-1, 1]
        kernel_xy = np.random.uniform(-1.0, 1.0, (kernel_size, 2)).astype(np.float32)
        kernel_z = np.random.uniform(0.0, 1.0, (kernel_size, 1)).astype(np.float32)
        kernel = np.hstack((kernel_xy, kernel_z))
        
        # Normalize the vectors to fit inside a unit sphere
        kernel = kernel / np.linalg.norm(kernel, axis=1, keepdims=True)

        # Apply an acceleration function to push more points towards the center
        for i in range(kernel_size):
            scale = float(i) / float(kernel_size)
            # Use a quadratic function to concentrate sample points around the origin
            acceleration = 0.1 + 0.9 * scale * scale
            kernel[i] *= acceleration
        
        return kernel
    
    def generateSSAOKernelNoiseRotation(self, num=16):
        
        noise = np.random.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)
        noise[:, 2] = 0.0  # Set z component to 0
        return noise
    
    def generateNoiseTexture(self, w, h):

        noise = self.generateSSAOKernelNoiseRotation(num=w*h)

        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)

        # if (im.size/im_h)%4 == 0:
        #     glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        # else:
        #     glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, noise)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return tid
        
        


    def setEnableSSAO(self, enable=True):
        """
        Enable or disable SSAO (Screen Space Ambient Occlusion).
        Args:
            enable (bool): True to enable SSAO, False to disable.
        Returns:
            None
        """
        self.enableSSAO = 1 if enable else 0
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

        
        # self.resetCamera()
        
        self.grid.manualBuild()
        self.smallGrid.manualBuild()
        self.axis.manualBuild()

        self.fullScreenQuad = FullScreenQuad()
        print('Compiling OpenGL shaders...')


        if sys.platform == 'darwin':
            print('Using OpenGL 1.2')
            _version = '120'
            
        else:
            print('OpenGL version: 3.3')            
            _version = '330'

        # _version = '120'
        self.SSAOGeoProg = self.buildShader(
            vshader_path=f'./glw/shaders/{_version}/ssao_geo_vs.glsl',
            fshader_path=f'./glw/shaders/{_version}/ssao_geo_fs.glsl'
        )
        self.SSAOCoreProg = self.buildShader(
            vshader_path=f'./glw/shaders/{_version}/ssao_core_vs.glsl',
            fshader_path=f'./glw/shaders/{_version}/ssao_core_fs.glsl'
        )
        self.SSAOBlurProg = self.buildShader(
            vshader_path=f'./glw/shaders/{_version}/ssao_blur_vs.glsl',
            fshader_path=f'./glw/shaders/{_version}/ssao_blur_fs.glsl'
        )
        self.SSAOLightProg = self.buildShader(
            vshader_path=f'./glw/shaders/{_version}/ssao_light_vs.glsl',
            fshader_path=f'./glw/shaders/{_version}/ssao_light_fs.glsl'
        )
             

        self.geoProgAttribList = ['a_Position', 'a_Normal']
        self.geoProgUniformList = ['u_ProjMatrix', 'u_ViewMatrix', 'u_ModelMatrix']

        self.coreBlurProgAttribList = ['a_Position']
        
        self.coreProgUniformList = ['u_projMode', 'u_screenSize', 'u_kernelNoise', 'u_normalMap', 'u_positionMap', 'u_ProjMatrix', 'u_kernelSize', 'u_sampleRad', 'u_kernel']
        self.blurProgUniformList = ['u_AOMap', 'u_TexelSize', 'u_NormalMap', 'u_PositionMap', 'u_Radius', 'u_NormalSigma', 'u_DepthSigma', 'u_SpatialSigma']
        self.lightProgAttribList = ['a_Position', 'a_Color', 'a_Normal', 'a_Texcoord']
        self.lightProgUniformList = ['u_ProjMatrix', 'u_ViewMatrix', 'u_ModelMatrix', 'u_CamPos', 'u_AOMap', 'u_enableAO', \
                                'u_LightDir', 'u_LightColor', 'u_AmbientColor', 'u_Shiny', 'u_Specular', 'u_Diffuse', 'u_Pellucid', 'u_NumLights', \
                                'u_Lights[0].position', 'u_Lights[0].color', \
                                'u_Lights[1].position', 'u_Lights[1].color', \
                                'u_Lights[2].position', 'u_Lights[2].color', \
                                'u_Lights[3].position', 'u_Lights[3].color', \
                                'u_Lights[4].position', 'u_Lights[4].color', \
                                    'u_Texture','render_mode',
                                    'u_farPlane',
                                    'u_farPlane_ratio',
                                    'u_screenSize',
        ]

        print('Shaders compiled successfully.')

        self.SSAOGeoProgLocMap = self.cacheShaderLocMap(self.SSAOGeoProg, self.geoProgAttribList, self.geoProgUniformList)
        self.SSAOLightProgLocMap = self.cacheShaderLocMap(self.SSAOLightProg, self.lightProgAttribList, self.lightProgUniformList)
        
        self.SSAOCoreProgLocMap = self.cacheShaderLocMap(self.SSAOCoreProg, self.coreBlurProgAttribList, self.coreProgUniformList)
        self.SSAOBlurProgLocMap = self.cacheShaderLocMap(self.SSAOBlurProg, self.coreBlurProgAttribList, self.blurProgUniformList)

        self.SSAOGeoFBO = FBOManager()
        self.SSAOCoreFBO = FBOManager()
        self.SSAOBlurFBO = FBOManager()

        self.SSAONoiseTexture = self.generateNoiseTexture(4, 4)

        # setup SSAO core shaders

        kernelSize = 64
        kernelLength = 1.0
        kernel = self.generateSSAOKernel(kernelSize) * kernelLength
        glUseProgram(self.SSAOCoreProg)
        
        glUniform3fv(self.SSAOCoreProgLocMap['u_kernel'], kernelSize, kernel.flatten())
        glUniform1f(self.SSAOCoreProgLocMap['u_sampleRad'], kernelLength)
        glUniform1i(self.SSAOCoreProgLocMap['u_kernelSize'], kernelSize)

        glUseProgram(self.SSAOBlurProg)

        glUniform1f(self.SSAOBlurProgLocMap["u_SpatialSigma"], 2.0)
        glUniform1f(self.SSAOBlurProgLocMap["u_DepthSigma"], 0.5)
        glUniform1f(self.SSAOBlurProgLocMap["u_NormalSigma"], 32.0)
        glUniform1i(self.SSAOBlurProgLocMap["u_Radius"], 2)

        # setup SSAO lighting shaders
        
        glUseProgram(self.SSAOLightProg)
        
        glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[0].position'), *self.key_light_pos)
        glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[0].color'),    *self.key_light_color)
        glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[1].position'), *self.fill_light_pos)
        glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[1].color'),    *self.fill_light_color)
        glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[2].position'), *self.back_light_pos)
        glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[2].color'),    *self.back_light_color)
        # glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[3].position'), *self.bottom_light_pos)
        # glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[3].color'),    *self.bottom_light_color)
        # glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[4].position'), *self.top_light_pos)
        # glUniform3f(self.SSAOLightProgLocMap.get('u_Lights[4].color'),    *self.top_light_color)
        
        glUniform1i(self.SSAOLightProgLocMap.get('u_NumLights'), 3)
        glUniform3f(self.SSAOLightProgLocMap['u_AmbientColor'], *self.ambient)
        glUniform1f(self.SSAOLightProgLocMap['u_Shiny'], self.shiny)
        glUniform1f(self.SSAOLightProgLocMap['u_Specular'], self.specular)
        glUniform1f(self.SSAOLightProgLocMap['u_Diffuse'], self.diffuse)
        glUniform1f(self.SSAOLightProgLocMap['u_Pellucid'], self.pellucid)

        glUseProgram(0)

        

    def _tempRenderFullScreenQuad(self):
        '''
        Renders a full-screen quad.
        '''
        glBegin(GL_QUADS)
    
        glVertex3f(-1.0, -1.0, 0.0)
        glVertex3f(1.0, -1.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)
        glVertex3f(-1.0, 1.0, 0.0)
      
        glEnd()

    def _renderObjs(self, locMap:dict, render_mode:int, size:float):
        '''
        A helper function to render all objects in the scene.
        Args:
            locMap (dict): The location map for shader variables.
            render_mode (int): The rendering mode (points, lines, or faces).
            size (float): The size of the points/lines.
        Returns:
            None
        '''
        for k, v in self.objectList.items():
            if hasattr(v, 'renderinShader'):
                v.renderinShader(locMap=locMap, render_mode=render_mode, size=size)


    def _copyBuffer2Screen(self, buffer:FBOManager):
        '''
        Copy the contents of the specified framebuffer object to the screen.
        Note: requires GL_RGBA
        Args:
            buffer (FBOManager): The framebuffer object to copy from.
        '''
        
        glBindFramebuffer(GL_READ_FRAMEBUFFER, buffer._fbo)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())
        glDrawBuffer(GL_COLOR_ATTACHMENT0)
        
        glBlitFramebuffer(
            0, 0, self.raw_window_w, self.raw_window_h,
            0, 0, self.raw_window_w, self.raw_window_h,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR
        )
        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        

    def paintGL(self):
        
        
        self.camera.setAspectRatio(float(self.scaled_window_w) / float(self.scaled_window_h))
        projMatrix = self.camera.updateProjTransform(isEmit=False)
        camtrans = self.camera.updateTransform(isEmit=False)
        campos = np.linalg.inv(camtrans)[:3,3]        


        ''' stage 1: SSAO Geometry Pass'''

        self.SSAOGeoFBO.getFBO(self.raw_window_w, self.raw_window_h, depth=True, colors=[GL_RGBA32F, GL_RGBA32F])
        self.SSAOGeoFBO.bindForWriting()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.SSAOGeoProg)
        
        # Set all matrixs
        glUniformMatrix4fv(self.SSAOGeoProgLocMap['u_ModelMatrix'], 1, GL_FALSE, self.canonicalModelMatrix, None)
        glUniformMatrix4fv(self.SSAOGeoProgLocMap['u_ProjMatrix'],  1, GL_FALSE, projMatrix, None)
        glUniformMatrix4fv(self.SSAOGeoProgLocMap['u_ViewMatrix'],  1, GL_FALSE, camtrans.T, None)

        # render objs
        self._renderObjs(locMap=self.SSAOGeoProgLocMap, render_mode=self.gl_render_mode, size=self.point_line_size)


        ''' stage 2: SSAO Core Pass '''
        if self.enableSSAO:
            
            self.SSAOCoreFBO.getFBO(self.raw_window_w, self.raw_window_h, depth=False, colors=[GL_RGBA32F])
            self.SSAOCoreFBO.bindForWriting()
            glClear(GL_COLOR_BUFFER_BIT)
            
            glUseProgram(self.SSAOCoreProg)


            glUniformMatrix4fv(self.SSAOCoreProgLocMap['u_ProjMatrix'], 1, GL_FALSE, projMatrix, None)

            self.SSAOGeoFBO.bindTextureForReading(GL_TEXTURE21, 1)
            glUniform1i(self.SSAOCoreProgLocMap['u_positionMap'], 21)

            self.SSAOGeoFBO.bindTextureForReading(GL_TEXTURE22, 2)
            glUniform1i(self.SSAOCoreProgLocMap['u_normalMap'], 22)

            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.SSAONoiseTexture)
            glUniform1i(self.SSAOCoreProgLocMap['u_kernelNoise'], 3)
            glUniform2f(self.SSAOCoreProgLocMap['u_screenSize'], float(self.raw_window_w), float(self.raw_window_h))
            glUniform1i(self.SSAOCoreProgLocMap['u_projMode'], 
                        0 if self.camera.projection_mode == GLCamera.projectionMode.perspective else 1)

            self._tempRenderFullScreenQuad()
            


            ''' stage 3: SSAO Blur Pass '''

            self.SSAOBlurFBO.getFBO(self.raw_window_w, self.raw_window_h, depth=False, colors=[GL_RGBA32F])
            self.SSAOBlurFBO.bindForWriting()
            glClear(GL_COLOR_BUFFER_BIT)

            glUseProgram(self.SSAOBlurProg)

            self.SSAOCoreFBO.bindTextureForReading(GL_TEXTURE21, 0)
            glUniform1i(self.SSAOBlurProgLocMap["u_AOMap"], 21)

            self.SSAOGeoFBO.bindTextureForReading(GL_TEXTURE22, 1)
            glUniform1i(self.SSAOBlurProgLocMap["u_PositionMap"], 22)
            self.SSAOGeoFBO.bindTextureForReading(GL_TEXTURE23, 2)
            glUniform1i(self.SSAOBlurProgLocMap["u_NormalMap"], 23)

            glUniform2f(self.SSAOBlurProgLocMap["u_TexelSize"],
                        1.0 / float(self.raw_window_w),
                        1.0 / float(self.raw_window_h))



            self._tempRenderFullScreenQuad()


        ''' stage 4: SSAO Lighting Pass '''

        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.SSAOLightProg)
        
        if self.enableSSAO:
            self.SSAOBlurFBO.bindTextureForReading(GL_TEXTURE21, 0)
            glUniform1i(self.SSAOLightProgLocMap['u_AOMap'], 21)

        glUniform1i(self.SSAOLightProgLocMap['u_enableAO'], self.enableSSAO)
        glUniform3f(self.SSAOLightProgLocMap['u_CamPos'], *campos)

        glUniformMatrix4fv(self.SSAOLightProgLocMap['u_ModelMatrix'], 1, GL_FALSE, self.canonicalModelMatrix, None)
        glUniformMatrix4fv(self.SSAOLightProgLocMap['u_ProjMatrix'],  1, GL_FALSE, projMatrix, None)
        glUniformMatrix4fv(self.SSAOLightProgLocMap['u_ViewMatrix'],  1, GL_FALSE, camtrans.T, None)

        
        glUniform2f(self.SSAOLightProgLocMap['u_screenSize'], float(self.raw_window_w), float(self.raw_window_h))

        self._renderObjs(locMap=self.SSAOLightProgLocMap, render_mode=self.gl_render_mode, size=self.point_line_size)
        
        
        # stage Final: Copy framebuffer to screen (default) framebuffer
        # NOTE: remove before flight
        
        # self._copyBuffer2Screen(self.SSAOBlurFBO)
        
        
        
        glDepthMask(GL_FALSE)
        
        if self.isAxisVisable:
            self.axis.renderinShader(locMap=self.SSAOLightProgLocMap)
            
        if self.isGridVisable:
            glUniform1i(self.SSAOLightProgLocMap['u_farPlane'], 1)
            glUniform1f(self.SSAOLightProgLocMap['u_farPlane_ratio'], 0.02)

            self.grid.renderinShader(locMap=self.SSAOLightProgLocMap)
            glUniform1f(self.SSAOLightProgLocMap['u_farPlane_ratio'], 0.15)
            self.smallGrid.renderinShader(locMap=self.SSAOLightProgLocMap)
            glUniform1i(self.SSAOLightProgLocMap['u_farPlane'], 0)

        glDepthMask(GL_TRUE)

        
        glFlush()



    def reset(self, ):
        
        for k, v in self.objectList.items():
            if hasattr(v, 'reset'):
                v.reset()
        self.objectList = {}
        
        self.update()
        

    def resizeGL(self, w: int, h: int) -> None:
        self.scaled_window_w = w
        self.scaled_window_h = h

        self.PixelRatio = self.devicePixelRatioF()
        
        self.raw_window_w = int(w * self.PixelRatio)
        self.raw_window_h = int(h * self.PixelRatio)

        self.camera.updateIntr(self.raw_window_h, self.raw_window_w)
        
        self.statusbar.move(0, h-self.statusbar.height())
        self.statusbar.resize(w, h)

        self.gl_settings.move((self.scaled_window_w - self.gl_setting_button.width()) - 20, 15)
        
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
        projected_coordinates[0] = (self.scaled_window_w * self.PixelRatio) - projected_coordinates[0]
        return int(projected_coordinates[1]//self.PixelRatio), int(projected_coordinates[0]//self.PixelRatio)
    
    
    def UVtoWorldCoordinate(self, u, v, dis=10):
        '''
        Convert UV coordinates to 3D world coordinates.
        u, v: int
        dis: float, distance from camera
        '''
        camCoord = self.camera.rayVector(u, v, dis=dis)
        p = camCoord[:3] / camCoord[3]
        print(f'UV to 3D: {u}, {v} -> {p}')
        return p


    def worldDrawText(self, p, h=18, w=0, msg='', txc='#EEEEEE', bgc='2e68c5', lenratio=8):

        v, u = self.worldCoordinatetoUV(p)

        glDisable(GL_DEPTH_TEST)
        self.textPainter.begin(self)

        if not w:
            w = len(msg) * lenratio
        
        self.textPainter.setViewport(0, 0, self.scaled_window_w * self.PixelRatio, self.scaled_window_h * self.PixelRatio)
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
                
        self.update()
        
        self.getDepthMap()
        
        mouseCoordinateinViewPortX = int((self.lastPos.x()) * self.PixelRatio )
        mouseCoordinateinViewPortY = int((self.scaled_window_h -  self.lastPos.y()) * self.PixelRatio)

        self.mouseClickPointinUV = np.array([mouseCoordinateinViewPortX, mouseCoordinateinViewPortY])
        
        depth_value = self.depthMap[int(self.lastPos.y() * self.PixelRatio), mouseCoordinateinViewPortX] if self.depthMap is not None else self.camera.far
        liner_depth_value = DepthReader.convertNDC2Liner(depth_value, self.camera)
        # print(f'Depth value at ({mouseCoordinateinViewPortX}, {mouseCoordinateinViewPortY}): {liner_depth_value}')

        self.mouseClickPointinWorldCoordinate = self.camera.rayVector(mouseCoordinateinViewPortX, mouseCoordinateinViewPortY, dis=liner_depth_value)
        
        # if event.buttons() & Qt.RightButton:
        #     transform = np.identity(4, dtype=np.float32)
        #     transform[:3, 3] = self.mouseClickPointinWorldCoordinate[:3]
        #     self.updateObject(ID=1, obj=Axis(
        #         transform=transform,
        #     ))
            # print('campose', self.camera.CameraTransformMat)
        if event.buttons() & Qt.RightButton:
            self.rightMouseClickSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        elif event.buttons() & Qt.MiddleButton:
            self.middleMouseClickSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        elif event.buttons() & Qt.LeftButton:
            self.leftMouseClickSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)

        # self.update()

    def mouseMoveEvent(self, event:QMouseEvent):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        

        if event.buttons() & Qt.LeftButton:
                    
            if self.camera.controltype == self.camera.controlType.arcball:
                # archball rotation
                self.camera.rotate(
                    [event.x(), event.y()],
                    [self.lastPos.x(), self.lastPos.y()],
                    self.scaled_window_h,
                    self.scaled_window_w
                )
            else:
                # Fix up rotation
                self.camera.rotate(dx, dy)

        if event.buttons() & Qt.RightButton:
                        
            self.camera.translate(dx, dy)

        self.lastPos = event.pos()

        self.mouseMoveSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)

        self.triggerFlush()

    def wheelEvent(self, event:QWheelEvent):
        angle = event.angleDelta()
            
        self.camera.zoom(angle.y()/200.)
        
        self.camera.updateIntr(self.raw_window_h, self.raw_window_w)

        self.triggerFlush()

    def mouseDoubleClickEvent(self, event:QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)
        
        if event.buttons() & Qt.LeftButton:
            self.resetCamera()

        self.triggerFlush()
        
    def mouseReleaseEvent(self, event):
        self.mouseReleaseSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        return super().mouseReleaseEvent(event)
    
    
    def getDepthMap(self):
        
        self.SSAOGeoFBO.bindForWriting()
        depth_data = glReadPixels(0, 0, self.raw_window_w, self.raw_window_h,
                                GL_DEPTH_COMPONENT, GL_FLOAT)
        
        depth_array = np.frombuffer(depth_data, dtype=np.float32)
        self.depthMap = depth_array.reshape((self.raw_window_h, self.raw_window_w))[::-1, :]

        liner_depth = DepthReader.convertNDC2Liner(self.depthMap, self.camera)

        # glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())

        return liner_depth
    
    
    def saveDepthMap(self, path=None):
        # if self.depthMap is not None:
        liner_depth = self.getDepthMap()
        depth_image = liner_depth.astype(np.uint16)
        depth_image_pil = Image.fromarray(depth_image, mode='I;16')

        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Save Depth Map', './depth.png', 'PNG Files (*.png);;All Files (*)')

        if path:
            depth_image_pil.save(path)
            print(f'Depth map saved to {path}')

            
    def saveRGBAMap(self, path=None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Save RGBA Image', './image.png', 'PNG Files (*.png);;All Files (*)')
        
        if path:
            image = self.grabFramebuffer()
            image.save(path)
            print(f'RGBA image saved to {path}')
        else:
            print('No path specified to save RGBA image.')
