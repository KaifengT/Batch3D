'''
copyright: (c) 2025 by KaifengTang, TingruiGuo
'''
import sys, os
import traceback
import numpy as np
from PySide6.QtCore import (Qt, Signal, QPoint, QTimer)
from PySide6.QtGui import (QColor, QWheelEvent, QMouseEvent, QSurfaceFormat, QFont)
from PySide6.QtWidgets import (QWidget, QFileDialog)
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from typing import Tuple, Iterable, Optional, Union
from .GLMesh import Mesh, PointCloud, Grid, Axis, BoundingBox, Lines, Arrow, BaseObject, FullScreenQuad, Sphere, UnionObject

from PIL import Image
from .GLCamera import GLCamera
from .GLMenu import GLSettingWidget

# from memory_profiler import profile
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
        elif internalType == GL_RED:
            return GL_RED, GL_FLOAT
        else:
            raise ValueError(f"Unsupported internal type: {internalType}")

    def getFBO(self, width:int, height:int, depth:bool=False, ms:bool=False, samples:int=1, colors:Iterable[int]=[]) -> Tuple[int, int]:
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
            self._createFBO(width, height, depth, ms, samples, colors)
        # self._create_fbo(width, height)
        return self._fbo, self._depth_texture

    def _addDepthAttachment(self, width:int, height:int):
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

    def _addDepthAttachmentMultisample(self, width:int, height:int, samples:int=1):
        '''
        Add a depth attachment to the FBO.
        Args:
            width (int): The width of the FBO.
            height (int): The height of the FBO.
        Returns:
            None
        '''
        depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, depth_texture)
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_DEPTH_COMPONENT32,
                                 width, height, True)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                GL_TEXTURE_2D_MULTISAMPLE, depth_texture, 0)
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
        self._attachments_id.append(depth_texture)


    def _addAttachment(self, width:int, height:int, internalType:int, attachment:int=GL_COLOR_ATTACHMENT0, filter:int=GL_NEAREST):
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


    def _addAttachmentMultisample(self, width:int, height:int, internalType:int, attachment:int=GL_COLOR_ATTACHMENT0, filter:int=GL_NEAREST, samples:int=1):
        '''
        Add a multisampled color attachment to the FBO.
        Args:
            internalType (int): The internal format of the texture.
                supported types: GL_RGBA32F, GL_RGB32F, GL_R32F, GL_RGB
            attachment (int): The attachment point of the texture. Choose from GL_COLOR_ATTACHMENT0 - GL_COLOR_ATTACHMENT31.
            filter (int): The texture filter mode.
                supported types: GL_LINEAR, GL_NEAREST
            samples (int): The number of samples to use for multisampling.

        Returns:
            None
        '''
        
        texID = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texID)
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, internalType,
                                 width, height, True)

        # glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MIN_FILTER, filter)
        # glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MAG_FILTER, filter)
        # glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        # glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment,
                            GL_TEXTURE_2D_MULTISAMPLE, texID, 0)

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)

        self._attachments_id.append(texID)

    def _createFBO(self, width:int, height:int, depth:bool=False, ms:bool=False, samples:int=1, colors:Iterable[int]=[]):

        # print(f"FBOManager: Creating FBO: {width}x{height}")
        
        if self._fbo is not None:
            # print(f"FBOManager: Cleaning up existing FBO and attachments: {self._fbo}")
            self.cleanUp()
        

        self._fbo = glGenFramebuffers(1)
        # print(f"FBOManager: Generated FBO ID: {self._fbo}")
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # depth attachment, necessary.
        if depth:
            if ms:
                self._addDepthAttachmentMultisample(width, height, samples=samples)
            else:
                self._addDepthAttachment(width, height)

        for i, iType in enumerate(colors):
            if ms:
                self._addAttachmentMultisample(width, height, iType, attachment=GL_COLOR_ATTACHMENT0 + i, filter=GL_NEAREST, samples=samples)
            else:
                self._addAttachment(width, height, iType, attachment=GL_COLOR_ATTACHMENT0 + i, filter=GL_LINEAR)

        glDrawBuffers(len(colors), [GL_COLOR_ATTACHMENT0 + i for i in range(len(colors))])

                
        # check if the framebuffer is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print(f'FBOManager: FBO creation failed: {glCheckFramebufferStatus(GL_FRAMEBUFFER)}')
            exit(0)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._width = width
        self._height = height
        
    def bindForWriting(self, ):
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


    def bindTextureForReading(self, textureUnit:int, attachmentIndex:int):
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
        
            
    def cleanUp(self, ):
        '''
        cleanup the FBO and its associated textures.
        '''
        # print(f'FBOManager: trying to cleanup FBO resources {self._fbo}, textures {self._attachments_id}')
        try:
            if len(self._attachments_id):
                glDeleteTextures(len(self._attachments_id), self._attachments_id)
                self._attachments_id = []
        except Exception as e:
            print(f"FBOManager: error occurred while cleaning texture resources: {e}")

        try:
            if self._fbo is not None:
                glDeleteFramebuffers(1, [self._fbo])
                self._fbo = None
        except Exception as e:
            print(f"FBOManager: error occurred while cleaning FBO resources: {e}")


    # def __del__(self):
    #     print('calling __del__')
    #     self.cleanUp()

class DepthReader:
        
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
    
    
class PointLight:
    def __init__(self, position:np.ndarray, color:np.ndarray, intensity:float=1.0) -> None:
        self.position = position
        self.color = color
        self.intensity = intensity


class GLWidget(QOpenGLWidget):

    # NOTE: these signals should not be used for internal communication
    leftMouseClickSignal = Signal(np.ndarray, np.ndarray)
    rightMouseClickSignal = Signal(np.ndarray, np.ndarray)
    middleMouseClickSignal = Signal(np.ndarray, np.ndarray)
    mouseReleaseSignal = Signal(np.ndarray, np.ndarray)
    mouseMoveSignal = Signal(np.ndarray, np.ndarray)

    # NOTE: signals belows are used for internal communication
    infoSignal = Signal(str, str, str) # title, message, type

    def __init__(self, 
        parent:Optional[QWidget]=None,
        backgroundColor:Tuple=(0, 0, 0, 0),
        **kwargs,
        ) -> None:

        super().__init__(parent)

        # For Windows
        if sys.platform == 'win32':
            backgroundColor = [0., 0., 0., 0.]
            self.font = QFont([u'Cascadia Mono', u'Microsoft Yahei UI'], 9, )
            # majorVersion = 4
            # minorVersion = 6
        # For macOS
        elif sys.platform == 'darwin':
            backgroundColor = [0.109, 0.117, 0.125, 1.0]
            self.font = QFont(['SF Pro Display', 'Helvetica Neue', 'Arial'], 10, QFont.Weight.Normal)
            # majorVersion = 1
            # minorVersion = 2
            
        else:
            backgroundColor = [0.109, 0.117, 0.125, 1.0]
            self.font = QFont([u'Cascadia Mono', u'Microsoft Yahei UI'], 9, )
            # majorVersion = 4
            # minorVersion = 6
            
            
        self.setMinimumSize(200, 200)            
            
        self._scaledWindowW = 0
        self._scaledWindowH = 0
        self._rawWindowW = 0
        self._rawWindowH = 0
        
        self._bgColor = backgroundColor
        self._objectList: dict[str, BaseObject] = {}
        self._lastPos = QPoint(0, 0)
        
        self._axisScale = 1.0
        self._isAxisVisable = True
        self._isGridVisable = True
        self._glRenderMode = 3
        
        self._enableSSAO = 1
        self._SSAOkernelSize = 64
        self._SSAOStrength = 60.0
        
        
        # GLFormat = self.format()
        # GLFormat.setVersion(majorVersion, minorVersion)
        # GLFormat.setProfile(QSurfaceFormat.CoreProfile)
        # GLFormat.setSamples(4)  # 4x MSAA
        # GLFormat.setSwapInterval(1)
        # self.setFormat(GLFormat)


        self.mouseClickPointinWorldCoordinate = np.array([0,0,0,1])
        self.mouseClickPointinUV = np.array([0, 0])

        self.canonicalModelMatrix = np.identity(4, dtype=np.float32)
                        
        self.camera = GLCamera()
        self.camera.updateSignal.connect(self.update)
        

        self.keyLightPos = np.array([0.0, 1.1, 1.1], dtype=np.float32) * 10000
        self.keyLightColor = np.array([0.3, 0.3, 0.3], dtype=np.float32)

        self.fillLightPos = np.array([-1.0, -1.2, -1.2], dtype=np.float32) * 10000
        self.fillLightColor = np.array([0.3, 0.4, 0.4], dtype=np.float32)

        self.backLightPos = np.array([1.0, -0.9, 1.3], dtype=np.float32) * 10000
        self.backLightColor = np.array([0.4, 0.4, 0.3], dtype=np.float32)

        self.defaultLights:list[PointLight] = [
            PointLight(self.keyLightPos, self.keyLightColor, 1.0),
            PointLight(self.fillLightPos, self.fillLightColor, 1.0),
            PointLight(self.backLightPos, self.backLightColor, 1.0)
        ]

        # self.topLightPos = np.array([0.2, 0.3, 1], dtype=np.float32) * 10000
        # self.topLightColor = np.array([0.2, 0.7, 0.2], dtype=np.float32)

        # self.bottomLightPos = np.array([0.4, 0.1, -1.2], dtype=np.float32) * 10000
        # self.bottomLightColor = np.array([0.7, 0.2, 0.2], dtype=np.float32)

        self.defaultAmbient = np.array([0.7, 0.7, 0.7], dtype=np.float32)  # 环境光颜色

        self.grid = Grid()
        self.smallGrid = Grid(n=510, scale=0.1)
        self.axis = Axis()
        
        self.glSettings = GLSettingWidget(
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
            ssao_kernel_size_callback=self.setSSAOKernelSize,
            ssao_strength_callback=self.setSSAOStrength,
        )
        
        self.glSettingButton = self.glSettings.get_button()
        self.glCameraPerpCombobox = self.glSettings.gl_camera_perp_combobox
        self.glCameraViewCombobox = self.glSettings.gl_camera_view_combobox

                
        self.FPSTimer = QTimer()
        self.FPSTimer.timeout.connect(self.countFPS)
        self.FPSTimer.setInterval(1000) # 1 second
        
        self._fps = 0
        
        self._lastSavePath = ''

    def enableCountFps(self, enable:bool=True):
        if enable:
            self._fps = 0
            self.FPSTimer.start(1000)
        else:
            self.FPSTimer.stop()

    def countFPS(self, ):
        print(self._fps)
        self._fps = 0

    def setBackgroundColor(self, color: Tuple[float, float, float, float]):
        '''
        This method sets the background color of the OpenGL widget.
        no need to use this func for Windows platform
        Args:
            color (tuple): A tuple of 4 floats representing the RGBA color values, each in the range 0-1.
        Returns:
            None
        '''
        assert len(color) == 4, "Color must be a tuple of 4 floats (R, G, B, A) in range 0-1."
        if all(0 <= c <= 1. for c in color):
            self._bgColor = color
            print(f'Setting background color to: {self._bgColor}')
            self.makeCurrent()
            glClearColor(*self._bgColor)
            self.update()
        else:
            raise ValueError("Color values must be in the range 0-1.")


    def setCameraControl(self, index:int):
        '''
        Set camera control type
        Args:
            index (int): The index of the camera control type.
             - 0: Arcball
             - 1: Orbit
        '''
        self.camera.controltype = self.camera.controlType(index)
        self.resetCamera()
        
    def setCameraPerspMode(self, index:int):
        '''
        Set camera perspective mode
        Args:
            index (int): The index of the camera perspective mode.
             - 0: Perspective
             - 1: Orthographic
        '''
        self.camera.setProjectionMode(self.camera.projectionMode(index))
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        self.update()
        
    def setAxisVisibility(self, isVisible:bool=True):
        '''
        Set axis visibility
        Args:
            isVisible (bool): Whether the axis should be visible or not.
        '''
        self._isAxisVisable = isVisible
        self.update()
        
    def setAxisScale(self, scale:float=1.0):
        '''
        Set axis size
        Args:
            scale (float): The scale factor for the axis.
        '''
        self._axisScale = scale
        scaledMatrix = np.identity(4, dtype=np.float32)
        scaledMatrix[:3,:3] *= self._axisScale
        self.axis.setTransform(scaledMatrix)
        self.update()
    
    def setGridVisibility(self, isVisible:bool=True):
        '''
        Set grid visibility
        Args:
            isVisible (bool): Whether the grid should be visible or not.
        '''
        self._isGridVisable = isVisible
        self.update()
    
    def resetCamera(self, ):
        self.camera.setCamera(azimuth=135, elevation=-55, distance=10, lookatPoint=np.array([0., 0., 0.,]))
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        self.camera.setLockRotate(False)
        self.glCameraViewCombobox.setCurrentItem('6')
        
        if hasattr(self, 'grid'):
            self.grid.setMode(5)
        if hasattr(self, 'smallGrid'):
            self.smallGrid.setMode(5)

    def setCameraViewPreset(self, preset:int=0):
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
            self.glCameraPerpCombobox.setCurrentItem('0')
            self.camera.setLockRotate(False)
            self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        else:
            self.camera.setViewPreset(preset)
            self.grid.setMode(preset)
            self.smallGrid.setMode(preset)
            self.camera.setProjectionMode(GLCamera.projectionMode.orthographic)
            self.glCameraPerpCombobox.setCurrentItem('1')
            self.camera.setLockRotate(True)
            self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
            
    def setObjectProps(self, ID:Union[int, str], props:dict):
        '''
        Setting the properties of an object in the objectList.
        Args:
            ID (Union[int, str]): The ID of the object in the objectList.
            props (dict): A dictionary containing the properties to be updated.
                Available properties include:
                - 'size': Size of the object (float).
                - 'isShow': Visibility of the object (boolean).
                - 'transform': Transformation matrix of the object (4x4 numpy array), same as the one used in setObjTransform.
        Returns:
            None
        '''
        
        _ID = str(ID)
        if _ID in self._objectList.keys():
            self._objectList[_ID].setMultiProp(props)

        self.update()

    def setObjTransform(self, ID:Union[int, str], transform:Optional[np.ndarray]=None) -> None:
        '''
        Setting the transformation matrix of an object in the objectList.
        Args:
            ID (Union[int, str]): The ID of the object in the objectList.
            transform (np.ndarray(4, 4)): The homogeneous transformation matrix to be set.
                If None, the transformation matrix will be set to the identity matrix.
        Returns:
            None
        '''
        _ID = str(ID)
        if _ID in self._objectList.keys():
            if transform is not None:
                self._objectList[_ID].setTransform(transform)
            else:
                self._objectList[_ID].setTransform(np.identity(4, dtype=np.float32))
        
        self.update()

    def getObjectList(self, ) -> dict[str, BaseObject]:
        '''
        Get the objects in the objectList.
        Returns:
            dict[str, BaseObject]: A dictionary containing the objects in the objectList.
        '''
        return self._objectList

    def updateObject(self, ID:Union[int, str], obj:Optional[BaseObject]=None) -> None:
        '''
        Update the object in the objectList with a new object or remove it if obj is None.
        Args:
            ID (Union[int, str]): The ID of the object in the objectList.
            obj (BaseObject): The new object to be set. 
                If None, the object which name matches the ID will be removed from the list.
        Returns:
            None
        '''
        
        self.makeCurrent()
        
        _ID = str(ID)
        if isinstance(obj, BaseObject):
            if obj.vao.getVAO() == 0:
                obj.load()
            self._objectList.update({_ID:obj})
            
        else:
            if _ID in self._objectList.keys():
                self._objectList.pop(_ID)
                
        self.update()

    def setRenderMode(self, mode:int):
        '''
        Set the rendering mode.
        Args:
            mode (int): The rendering mode to be set.
               - 0: Line rendering
               - 1: Simple rendering
               - 2: Normal rendering
               - 3: Texture rendering
               - 4: Ambient Occlusion rendering
        '''
        self._glRenderMode = mode
        self.update()

    def buildShader(self, vshader_path:str, fshader_path:str, gshader_path:Optional[str]=None, manualVersion:str='420 core') -> int:
        '''
        Compile and link the vertex and fragment shaders.
        Args:
            vshader_src (str): The source code PATH of the vertex shader.
            fshader_src (str): The source code PATH of the fragment shader.
            gshader_src (Optional[str]): The source code PATH of the geometry shader.
        Returns:
            program (int): The OpenGL program ID.
        '''
        try:
            self.makeCurrent()
            
            vshader_src = f'#version {manualVersion}\n' + open(vshader_path, encoding='utf-8').read()
            fshader_src = f'#version {manualVersion}\n' + open(fshader_path, encoding='utf-8').read()

            vshader = shaders.compileShader(vshader_src, GL_VERTEX_SHADER)
            fshader = shaders.compileShader(fshader_src, GL_FRAGMENT_SHADER)

            if gshader_path is not None:
                gshader_src = f'#version {manualVersion}\n' + open(gshader_path, encoding='utf-8').read()
                gshader = shaders.compileShader(gshader_src, GL_GEOMETRY_SHADER)
                program = shaders.compileProgram(vshader, gshader, fshader)
            else:
                program = shaders.compileProgram(vshader, fshader)
            return program
        
        except Exception as e:
            print(f"Error compiling/linking shaders: {e}")
            traceback.print_exc()
            return None


    def _cacheShaderLocMap(self, program:int, attribList:Iterable[str], uniformList:Iterable[str]) -> dict[str, int]:

        self.makeCurrent()
        shaderLocMap = {}
        for attrib in attribList:
            shaderLocMap.update({attrib:glGetAttribLocation(program, attrib)})

        for uniform in uniformList:
            shaderLocMap.update({uniform:glGetUniformLocation(program, uniform)})
        return shaderLocMap

    @staticmethod
    def generateSSAOKernel(kernel_size:int=64) -> np.ndarray:
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

    @staticmethod
    def generateSSAOKernelNoiseRotation(num:int=16) -> np.ndarray:

        noise = np.random.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)
        noise[:, 2] = 0.0  # Set z component to 0
        return noise

    def generateNoiseTexture(self, w:int, h:int) -> int:

        self.makeCurrent()
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
        self._enableSSAO = 1 if enable else 0
        self.update()

    def setSSAOKernelSize(self, size:int):
        """
        Set the SSAO kernel size.
        Args:
            size (int): The new kernel size.
        Returns:
            None
        """
        self._SSAOkernelSize = size

        if hasattr(self, 'SSAOCoreProg') and self.SSAOCoreProg is not None:
            self.makeCurrent()
            kernel = self.generateSSAOKernel(self._SSAOkernelSize)
            glUseProgram(self.SSAOCoreProg)
            glUniform3fv(self.SSAOCoreProgLocMap['u_kernel'], self._SSAOkernelSize, kernel.flatten())
            glUniform1i(self.SSAOCoreProgLocMap['u_kernelSize'], self._SSAOkernelSize)
            glUseProgram(0)
            
            self.update()

    def setSSAOStrength(self, strength:float):
        """
        Set the SSAO strength.
        Args:
            strength (float): The new SSAO strength.
        Returns:
            None
        """
        self._SSAOStrength = strength

        if hasattr(self, 'SSAOCoreProg') and self.SSAOCoreProg is not None:
            self.makeCurrent()
            glUseProgram(self.SSAOCoreProg)
            glUniform1f(self.SSAOCoreProgLocMap['u_radiusPixels'], self._SSAOStrength)
            glUseProgram(0)
            
            self.update()

    def setLights(self, lights:Optional[list[PointLight]]=None):
        """
        Set the point lights for the scene.
        Args:
            lights (list[PointLight]): The list of point lights to set.
        Returns:
            None
        """
        lights = lights if lights is not None else self.defaultLights
        numLights = min(len(lights), 5)
        
        if hasattr(self, 'SSAOLightProg') and self.SSAOLightProg is not None:
            self.makeCurrent()
            
            glUseProgram(self.SSAOLightProg)

            for i in range(numLights):
                glUniform3f(self.SSAOLightProgLocMap[f'u_Lights[{i}].position'], *lights[i].position)
                glUniform3f(self.SSAOLightProgLocMap[f'u_Lights[{i}].color'],    *lights[i].color)

            glUniform1i(self.SSAOLightProgLocMap['u_NumLights'], numLights)
            glUseProgram(0)
            self.update()


    def setAmbientColor(self, color:Optional[tuple]=None):
        """
        Set the ambient color for the scene.
        Args:
            color (tuple): The new ambient color (R, G, B).
        Returns:
            None
        """
        color = color if color is not None else self.defaultAmbient
        if hasattr(self, 'SSAOLightProg') and self.SSAOLightProg is not None:
            self.makeCurrent()
            glUseProgram(self.SSAOLightProg)
            glUniform3f(self.SSAOLightProgLocMap['u_AmbientColor'], *color)
            glUseProgram(0)

            self.update()

    def initializeGL(self):
        
        try:

            glMajorVersion = glGetIntegerv(GL_MAJOR_VERSION)
            glMinorVersion = glGetIntegerv(GL_MINOR_VERSION)
            gl_version = glGetString(GL_VERSION).decode('utf-8')
            glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')
            renderer = glGetString(GL_RENDERER).decode('utf-8')
            vendor = glGetString(GL_VENDOR).decode('utf-8')
            print(f'OpenGL version: {gl_version}, major: {glMajorVersion}, minor: {glMinorVersion}')
            print(f'GLSL version: {glsl_version}')
            print(f'OpenGL profile: {self.context().format().profile().name}')
            print(f'OpenGL renderer: {renderer}')
            print(f'OpenGL vendor: {vendor}')

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_PROGRAM_POINT_SIZE)
            
            # glEnable(GL_CULL_FACE)
            # glCullFace(GL_BACK)

            glClearColor(*self._bgColor)

            self.quad = FullScreenQuad()        
            self.grid.load()
            self.smallGrid.load()
            self.axis.load()
            
            
            print('Compiling OpenGL shaders...')
            shaderVersion = f'{glMajorVersion}{glMinorVersion}0 core'
            print(f'OpenGL shader version: {shaderVersion}')
            self.shaderFolder = '460'
            # _version = '120'
            self.SSAOGeoProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_geo_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_geo_fs.glsl',
                manualVersion=shaderVersion
            )
            self.SSAOCoreProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_core_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_core_fs.glsl',
                manualVersion=shaderVersion
            )
            self.SSAOBlurProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_blur_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_blur_fs.glsl',
                manualVersion=shaderVersion
            )
            self.SSAOLightProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_fs.glsl',
                manualVersion=shaderVersion
            )

            self.SSAOLightLineProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_line_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_fs.glsl',
                gshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_line_gs.glsl',
                manualVersion=shaderVersion
            )

            self.geoProgAttribList = ['a_Position', 'a_Normal']
            self.geoProgUniformList = ['u_pointSize', 'u_mvpMatrix', 'u_mvMatrix', 'u_normalMatrix']

            self.coreBlurProgAttribList = ['a_Position']
            
            self.coreProgUniformList = ['u_projMode', 'u_screenSize', 'u_kernelNoise', 'u_normalMap', 'u_positionMap', 'u_ProjMatrix', 'u_kernelSize', 'u_radiusPixels', 'u_kernel']
            self.blurProgUniformList = ['u_AOMap', 'u_TexelSize', 'u_NormalMap', 'u_PositionMap', 'u_Radius', 'u_NormalSigma', 'u_DepthSigma', 'u_SpatialSigma']
            self.lightProgAttribList = ['a_Position', 'a_Color', 'a_Normal', 'a_Texcoord']
            self.lightProgUniformList = ['u_mvpMatrix', 'u_normalMatrix', 'u_worldNormalMatrix', 'u_ModelMatrix', 'u_CamPos', 'u_AOMap', 'u_enableAO', \
                                    'u_LightDir', 'u_LightColor', 'u_AmbientColor', 'u_NumLights', \
                                    'u_Lights[0].position', 'u_Lights[0].color', \
                                    'u_Lights[1].position', 'u_Lights[1].color', \
                                    'u_Lights[2].position', 'u_Lights[2].color', \
                                    'u_Lights[3].position', 'u_Lights[3].color', \
                                    'u_Lights[4].position', 'u_Lights[4].color', \
                                        'u_renderMode', 
                                        'u_EnableAlbedoTexture', 'u_AlbedoTexture', 'u_Metallic', 'u_Roughness', 
                                        'u_EnableMetallicRoughnessTexture', 'u_MetallicRoughnessTexture',
                                        'u_farPlane',
                                        'u_farPlaneRatio',
                                        'u_screenSize',
                                        'u_pointSize','u_lineWidth',
            ]

            print('Shaders compiled successfully.')

            self.SSAOGeoProgLocMap = self._cacheShaderLocMap(self.SSAOGeoProg, self.geoProgAttribList, self.geoProgUniformList)
            self.SSAOLightProgLocMap = self._cacheShaderLocMap(self.SSAOLightProg, self.lightProgAttribList, self.lightProgUniformList)
            
            self.SSAOCoreProgLocMap = self._cacheShaderLocMap(self.SSAOCoreProg, self.coreBlurProgAttribList, self.coreProgUniformList)
            self.SSAOBlurProgLocMap = self._cacheShaderLocMap(self.SSAOBlurProg, self.coreBlurProgAttribList, self.blurProgUniformList)

            self.SSAOLightLineProgLocMap = self._cacheShaderLocMap(self.SSAOLightLineProg, self.lightProgAttribList, self.lightProgUniformList)

            self.SSAOGeoFBO = FBOManager()
            self.SSAOCoreFBO = FBOManager()
            self.SSAOBlurFBO = FBOManager()

            self.SSAONoiseTexture = self.generateNoiseTexture(4, 4)

            # setup SSAO core shaders

            
            kernel = self.generateSSAOKernel(self._SSAOkernelSize)
            glUseProgram(self.SSAOCoreProg)
            
            glUniform3fv(self.SSAOCoreProgLocMap['u_kernel'], self._SSAOkernelSize, kernel.flatten())
            glUniform1i(self.SSAOCoreProgLocMap['u_kernelSize'], self._SSAOkernelSize)
            glUniform1f(self.SSAOCoreProgLocMap['u_radiusPixels'], self._SSAOStrength)
            glUseProgram(self.SSAOBlurProg)

            glUniform1f(self.SSAOBlurProgLocMap["u_SpatialSigma"], 2.0)
            glUniform1f(self.SSAOBlurProgLocMap["u_DepthSigma"], 0.5)
            glUniform1f(self.SSAOBlurProgLocMap["u_NormalSigma"], 32.0)
            glUniform1i(self.SSAOBlurProgLocMap["u_Radius"], 2)

            # setup SSAO lighting shaders
            
            glUseProgram(self.SSAOLightProg)

            self.setLights()
            self.setAmbientColor()


            glUseProgram(0)

        except Exception as e:
            traceback.print_exc()


    def _renderObjs(self, locMap:dict, viewMatrix:np.ndarray, projMatrix:np.ndarray):
        '''
        A helper function to render all objects in the scene.
        Args:
            locMap (dict): The location map for shader variables.
        '''
        for obj in self._objectList.values():
            self._setGeoProgMVPMatrix(locMap, obj.transform, viewMatrix, projMatrix)
            obj.render(locMap=locMap)

    def _setGeoProgMVPMatrix(self, locMap:dict, modelMatrix:np.ndarray, viewMatrix:np.ndarray, projMatrix:np.ndarray):
        '''
        Set the Model-View-Projection matrix for the SSAO geometry shader.
        Args:
            locMap (dict): The location map for shader variables.
            modelMatrix (np.ndarray): The model matrix.
            viewMatrix (np.ndarray): The view matrix.
            projMatrix (np.ndarray): The projection matrix.
        '''
        mvMatrix = viewMatrix @ modelMatrix
        mvpMatrix = projMatrix @ mvMatrix
        glUniformMatrix4fv(locMap['u_mvpMatrix'], 1, GL_FALSE, mvpMatrix.T, None)
        glUniformMatrix4fv(locMap['u_mvMatrix'], 1, GL_FALSE, mvMatrix.T, None)
        glUniformMatrix3fv(locMap['u_normalMatrix'], 1, GL_FALSE, np.linalg.inv(mvMatrix)[:3, :3], None)

    def _setLightProgMVPMatrix(self, locMap:dict, modelMatrix:np.ndarray, viewMatrix:np.ndarray, projMatrix:np.ndarray):
        '''
        Set the Model-View-Projection matrix for the SSAO lighting shader.
        Args:
            locMap (dict): The location map for shader variables.
            modelMatrix (np.ndarray): The model matrix.
            viewMatrix (np.ndarray): The view matrix.
            projMatrix (np.ndarray): The projection matrix.
        '''
        
        mvpMatrix = projMatrix @ viewMatrix @ modelMatrix
        glUniformMatrix4fv(locMap['u_ModelMatrix'], 1, GL_FALSE, modelMatrix.T, None)
        glUniformMatrix4fv(locMap['u_mvpMatrix'], 1, GL_FALSE, mvpMatrix.T, None)
        glUniformMatrix3fv(locMap['u_worldNormalMatrix'], 1, GL_FALSE, np.linalg.inv(modelMatrix)[:3, :3], None)
        
        

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
            0, 0, self._rawWindowW, self._rawWindowH,
            0, 0, self._rawWindowW, self._rawWindowH,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST
        )
        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _copyBuffer(self, src:FBOManager, dst:FBOManager, srcatt=GL_COLOR_ATTACHMENT0, dstatt=GL_COLOR_ATTACHMENT0):
        '''
        Copy the contents of the specified framebuffer object to another framebuffer object.
        Args:
            src (FBOManager): The source framebuffer object.
            dst (FBOManager): The destination framebuffer object.
            srcatt (GLenum): The color attachment to read from the source framebuffer.
            dstatt (GLenum): The color attachment to write to the destination framebuffer.
        '''
        glBindFramebuffer(GL_READ_FRAMEBUFFER, src._fbo)
        glReadBuffer(srcatt)

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst._fbo)
        glDrawBuffer(dstatt)

        glBlitFramebuffer(
            0, 0, self._rawWindowW, self._rawWindowH,
            0, 0, self._rawWindowW, self._rawWindowH,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST
        )

    def paintGL(self):
        
        
        self.camera.setAspectRatio(float(self._scaledWindowW) / float(self._scaledWindowH))
        projMatrix = self.camera.updateProjTransform(isEmit=False)
        camtrans = self.camera.updateTransform(isEmit=False)
        
        campos = np.linalg.inv(camtrans)[:3,3]
        
        if self._glRenderMode != 0:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  

        ''' stage 1: SSAO Geometry Pass'''

        self.SSAOGeoFBO.getFBO(self._rawWindowW, self._rawWindowH, depth=True, colors=[GL_RGB32F, GL_RGB32F])
        self.SSAOGeoFBO.bindForWriting()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.SSAOGeoProg)
        
        # Set all matrixs
        # glUniformMatrix4fv(self.SSAOGeoProgLocMap['u_ModelMatrix'], 1, GL_FALSE, self.canonicalModelMatrix, None)
        # glUniformMatrix4fv(self.SSAOGeoProgLocMap['u_ProjMatrix'],  1, GL_FALSE, projMatrix, None)
        # glUniformMatrix4fv(self.SSAOGeoProgLocMap['u_ViewMatrix'],  1, GL_FALSE, camtrans.T, None)

        # render objs
        self._renderObjs(locMap=self.SSAOGeoProgLocMap, viewMatrix=camtrans, projMatrix=projMatrix.T)

        ''' stage 2: SSAO Core Pass '''
        if self._enableSSAO:
            
            self.SSAOCoreFBO.getFBO(self._rawWindowW, self._rawWindowH, depth=False, colors=[GL_R32F])
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
            glUniform2f(self.SSAOCoreProgLocMap['u_screenSize'], float(self._rawWindowW), float(self._rawWindowH))
            glUniform1i(self.SSAOCoreProgLocMap['u_projMode'], 
                        0 if self.camera.projection_mode == GLCamera.projectionMode.perspective else 1)

            self.quad.render()
            
            

            ''' stage 3: SSAO Blur Pass '''

            self.SSAOBlurFBO.getFBO(self._rawWindowW, self._rawWindowH, depth=False, colors=[GL_R32F])
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
                        1.0 / float(self._rawWindowW),
                        1.0 / float(self._rawWindowH))



            self.quad.render()


        ''' stage 4: SSAO Lighting Pass '''

        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
      
        
        glUseProgram(self.SSAOLightProg)
        
        if self._enableSSAO:
            self.SSAOBlurFBO.bindTextureForReading(GL_TEXTURE21, 0)
            glUniform1i(self.SSAOLightProgLocMap['u_AOMap'], 21)

        glUniform1i(self.SSAOLightProgLocMap['u_enableAO'], self._enableSSAO)
        glUniform3f(self.SSAOLightProgLocMap['u_CamPos'], *campos)

        glUniform1i(self.SSAOLightProgLocMap['u_renderMode'], self._glRenderMode)
        glUniform2f(self.SSAOLightProgLocMap['u_screenSize'], float(self._rawWindowW), float(self._rawWindowH))

        
        glUseProgram(self.SSAOLightLineProg)

        glUniform1i(self.SSAOLightLineProgLocMap['u_enableAO'], 0)
        glUniform3f(self.SSAOLightLineProgLocMap['u_CamPos'], *campos)

        glUniform1i(self.SSAOLightLineProgLocMap['u_renderMode'], self._glRenderMode)
        glUniform2f(self.SSAOLightLineProgLocMap['u_screenSize'], float(self._rawWindowW), float(self._rawWindowH))




        for obj in self._objectList.values():
            if not isinstance(obj, UnionObject):
                if obj.renderType != GL_LINES:
                    glUseProgram(self.SSAOLightProg)
                    self._setLightProgMVPMatrix(self.SSAOLightProgLocMap, obj.transform, camtrans, projMatrix.T)
                    obj.render(locMap=self.SSAOLightProgLocMap)
                else:
                    glUseProgram(self.SSAOLightLineProg)
                    self._setLightProgMVPMatrix(self.SSAOLightLineProgLocMap, obj.transform, camtrans, projMatrix.T)
                    obj.render(locMap=self.SSAOLightLineProgLocMap)
            else:
                for _obj in obj.objs:
                    if _obj.renderType == GL_LINES:
                        glUseProgram(self.SSAOLightLineProg)
                        self._setLightProgMVPMatrix(self.SSAOLightLineProgLocMap, _obj.transform, camtrans, projMatrix.T)
                        _obj.render(locMap=self.SSAOLightLineProgLocMap)
                    else:
                        glUseProgram(self.SSAOLightProg)
                        self._setLightProgMVPMatrix(self.SSAOLightProgLocMap, _obj.transform, camtrans, projMatrix.T)
                        _obj.render(locMap=self.SSAOLightProgLocMap)

        # self._renderObjs(locMap=self.SSAOLightProgLocMap)
        
        
        # stage Final: Copy framebuffer to screen (default) framebuffer
        # NOTE: remove before flight
        
        # self._copyBuffer2Screen(self.SSAOBlurFBO)
        glUseProgram(self.SSAOLightLineProg)
        
        glDepthMask(GL_FALSE)
        
            
        if self._isGridVisable:
            glUniform1i(self.SSAOLightLineProgLocMap['u_farPlane'], 1)
            glUniform1f(self.SSAOLightLineProgLocMap['u_farPlaneRatio'], 0.02)
            self._setLightProgMVPMatrix(self.SSAOLightLineProgLocMap, self.grid.transform, camtrans, projMatrix.T)
            self.grid.render(locMap=self.SSAOLightLineProgLocMap)
            glUniform1f(self.SSAOLightLineProgLocMap['u_farPlaneRatio'], 0.15)
            self.smallGrid.render(locMap=self.SSAOLightLineProgLocMap)
            glUniform1i(self.SSAOLightLineProgLocMap['u_farPlane'], 0)
        if self._isAxisVisable:
            self._setLightProgMVPMatrix(self.SSAOLightLineProgLocMap, self.axis.transform, camtrans, projMatrix.T)
            self.axis.render(locMap=self.SSAOLightLineProgLocMap)


        glDepthMask(GL_TRUE)

        self._fps += 1

        glFlush()
        



    def reset(self, ):
        '''
        Clean all Object in the scene.
        '''
        
        # Each Object will change its context so we dont need to call makeContext()
        for k, v in self._objectList.items():
            if hasattr(v, 'cleanup'):
                v.cleanup()
        self._objectList = {}
        
        self.update()
        

    def resizeGL(self, w: int, h: int) -> None:
        self._scaledWindowW = w
        self._scaledWindowH = h

        self.pixelRatio = self.devicePixelRatioF()
        
        self._rawWindowW = int(w * self.pixelRatio)
        self._rawWindowH = int(h * self.pixelRatio)

        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        
        # self.statusbar.move(0, h-self.statusbar.height())
        # self.statusbar.resize(w, h)

        self.glSettings.move((self._scaledWindowW - self.glSettingButton.width()) - 20, 15)
        
        # print(f'GLWidget resized to {w}x{h}, PixelRatio: {self.PixelRatio}')
        return super().resizeGL(w, h)
        

    def worldCoordinatetoUV(self, p:np.ndarray) -> tuple[int, int]:
        '''
        Convert world coordinates to UV coordinates.
        Args:
            p (np.ndarray): The 4D point in world coordinates.

        Returns:
            uv (tuple): The UV coordinates.
        '''
        camCoord = self.camera.CameraTransformMat @ p  
        projected_coordinates = self.camera.intr @ camCoord[:3]
        projected_coordinates = projected_coordinates[:2] / projected_coordinates[2]
        projected_coordinates[0] = (self._scaledWindowW * self.pixelRatio) - projected_coordinates[0]
        return int(projected_coordinates[1]//self.pixelRatio), int(projected_coordinates[0]//self.pixelRatio)


    def UVtoWorldCoordinate(self, u:int, v:int, dis:float=10) -> np.ndarray:
        '''
        Convert UV coordinates to 3D world coordinates.
        Args:
            u (int): The u-coordinate in UV space.
            v (int): The v-coordinate in UV space.
            dis (float): The distance from the camera.
        Returns:
            p (np.ndarray): The 3D world coordinates.
        '''
        camCoord = self.camera.rayVector(u, v, dis=dis)
        p = camCoord[:3] / camCoord[3]
        print(f'UV to 3D: {u}, {v} -> {p}')
        return p
   
   
    def mousePressEvent(self, event:QMouseEvent):
        
        self._lastPos = event.pos()
        self.camera.updateTransform(isAnimated=True, isEmit=False)
        self.update()
        
        mouseCoordinateinViewPortX = int((self._lastPos.x()) * self.pixelRatio )
        mouseCoordinateinViewPortY = int((self._scaledWindowH -  self._lastPos.y()) * self.pixelRatio)
        mouseCoordinateinViewPortRY = int(self._lastPos.y() * self.pixelRatio)

        self.mouseClickPointinUV = np.array([mouseCoordinateinViewPortX, mouseCoordinateinViewPortY])
        linerDepthValue = self.getDepthPoint(mouseCoordinateinViewPortX, mouseCoordinateinViewPortY)[0]
        self.mouseClickPointinWorldCoordinate = self.camera.rayVector(mouseCoordinateinViewPortX, mouseCoordinateinViewPortY, dis=linerDepthValue)
        
        # if event.buttons() & Qt.RightButton:
        #     transform = np.identity(4, dtype=np.float32)
        #     transform[:3, 3] = self.mouseClickPointinWorldCoordinate[:3]
        #     self.updateObject(ID=1, obj=Axis(
        #         transform=transform,
        #     ))
        
        if event.buttons() & Qt.RightButton:
            self.rightMouseClickSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        elif event.buttons() & Qt.MiddleButton:
            self.middleMouseClickSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        elif event.buttons() & Qt.LeftButton:
            self.leftMouseClickSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)


    def mouseMoveEvent(self, event:QMouseEvent):
        dx = event.x() - self._lastPos.x()
        dy = event.y() - self._lastPos.y()
        
        # self.fps += 1

        if event.buttons() & Qt.LeftButton:
            if self.camera.controltype == self.camera.controlType.arcball:
                # archball rotation
                self.camera.rotate(
                    [event.x(), event.y()],
                    [self._lastPos.x(), self._lastPos.y()],
                    self._scaledWindowH,
                    self._scaledWindowW
                )
            else:
                # Fix up rotation
                self.camera.rotate(dx, dy)

        if event.buttons() & Qt.RightButton: 
            self.camera.translate(dx, dy)

        self._lastPos = event.pos()
        self.mouseMoveSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        self.update()

    def wheelEvent(self, event:QWheelEvent):
        
        angle = event.angleDelta()
        self.camera.zoom(angle.y()/200.)
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        self.update()

    def mouseDoubleClickEvent(self, event:QMouseEvent) -> None:
        
        super().mouseDoubleClickEvent(event)
        
        if event.buttons() & Qt.LeftButton:
            self.resetCamera()

        self.update()
        
    def mouseReleaseEvent(self, event:QMouseEvent):
        
        self.mouseReleaseSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        return super().mouseReleaseEvent(event)
    
    
    def getDepthMap(self, ) -> np.ndarray:
        '''
        Get the depth map from the framebuffer. Depth map is converted from NDC to linear space.
        Returns:
            linerDepth (np.ndarray): The linear depth map.
        '''
        
        self.makeCurrent()
        self.SSAOGeoFBO.bindForWriting()
        rawDepth = glReadPixels(0, 0, self._rawWindowW, self._rawWindowH,
                                GL_DEPTH_COMPONENT, GL_FLOAT)
        
        NDCDepth = np.frombuffer(rawDepth, dtype=np.float32).reshape((self._rawWindowH, self._rawWindowW))[::-1, :]
        linerDepth = DepthReader.convertNDC2Liner(NDCDepth, self.camera)

        return linerDepth


    def getDepthPoint(self, x:int, y:int) -> np.ndarray:
        '''
        Get the depth value at a specific pixel location.
        Args:
            x (int): The x-coordinate of the pixel.
            y (int): The y-coordinate of the pixel.
        Returns:
            linerDepth (np.ndarray): The linear depth value.
        '''
        
        self.makeCurrent()
        self.SSAOGeoFBO.bindForWriting()
        
        rawDepth = glReadPixels(x, y, 1, 1,
                                GL_DEPTH_COMPONENT, GL_FLOAT)
        
        NDCDepth = np.frombuffer(rawDepth, dtype=np.float32)
        NDCDepth = NDCDepth.flatten()

        linerDepth = DepthReader.convertNDC2Liner(NDCDepth, self.camera)

        return linerDepth


    def saveDepthMap(self, path:Optional[str]=None):
        '''
        Save the depth map to a file to the specified path.
        Args:
            path (Optional[str]): The file path to save the depth map.
        '''
        try:
            liner_depth = self.getDepthMap()
            depth_image = liner_depth.astype(np.uint16)
            depth_image_pil = Image.fromarray(depth_image, mode='I;16')

            if path is None:
                path, _ = QFileDialog.getSaveFileName(self, 
                                                      'Save Depth Map', 
                                                      os.path.join(self._lastSavePath, 'depth.png') if os.path.exists(self._lastSavePath) else './depth.png', 
                                                      'PNG Files (*.png);;All Files (*)')

            if path:
                self._lastSavePath = os.path.dirname(path)
                depth_image_pil.save(path)
                print(f'Depth map saved to {path}')
                self.infoSignal.emit('Depth Map Saved', f'Depth map saved to {path}', 'complete')
            else:
                print('No path specified to save depth map.')
                self.infoSignal.emit('Save Depth Map', 'No path specified to save depth map.', 'warning')
        except Exception as e:
            print(f'Error saving depth map: {e}')
            self.infoSignal.emit('Save Depth Map', f'Error saving depth map: {e}', 'error')

    def saveRGBAMap(self, path:Optional[str]=None):
        '''
        Save the RGBA image to a file to the specified path.
        Args:
            path (Optional[str]): The file path to save the RGBA image.
        '''
        try:
            if path is None:
                path, _ = QFileDialog.getSaveFileName(self, 
                                                      'Save RGBA Image',
                                                      os.path.join(self._lastSavePath, 'image.png') if os.path.exists(self._lastSavePath) else './image.png',
                                                      'PNG Files (*.png);;All Files (*)')
            
            if path:
                self._lastSavePath = os.path.dirname(path)
                image = self.grabFramebuffer()
                image.save(path)
                print(f'RGBA image saved to {path}')
                self.infoSignal.emit('RGBA Image Saved', f'RGBA image saved to {path}', 'complete')
            else:
                print('No path specified to save RGBA image.')
                self.infoSignal.emit('Save RGBA Image', 'No path specified to save RGBA image.', 'warning')
                
        except Exception as e:
            print(f'Error saving RGBA image: {e}')
            self.infoSignal.emit('Save RGBA Image', f'Error saving RGBA image: {e}', 'error')
            
            
    def __del__(self, ):
        try:
            self.reset()
        except:
            ...