'''
copyright: (c) 2025 by KaifengTang, TingruiGuo
'''
import sys, os
import traceback
import numpy as np
from PySide6.QtCore import (Qt, Signal, QPoint, QTimer, QByteArray, QBuffer, QIODevice, QMimeData)
from PySide6.QtGui import (QColor, QWheelEvent, QMouseEvent, QSurfaceFormat, QFont, QOpenGLContext, QImage)
from PySide6.QtWidgets import (QWidget, QFileDialog, QApplication)
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from typing import Tuple, Iterable, Optional, Union
from .GLMesh import Mesh, PointCloud, Grid, Axis, BoundingBox, Lines, Arrow, BaseObject, FullScreenQuad, Sphere, UnionObject, Label, Character

try:
    from OpenGL.GL import glBlendFunci, glBlendFuncSeparatei
except ImportError:
    try:
        from OpenGL.GL.ARB.draw_buffers_blend import glBlendFunciARB as glBlendFunci
        from OpenGL.GL.ARB.draw_buffers_blend import glBlendFuncSeparateiARB as glBlendFuncSeparatei
    except ImportError:
        pass

from PIL import Image
from .GLCamera import GLCamera
from .GLMenu import GLSettingWidget, getCameraComboBox

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
        self._has_depth = False
        self._is_multisample = False
        self._samples = 1
        self._colors = tuple()

    @staticmethod
    def getFormat(internalType):
        if internalType == GL_RGBA32F:
            return GL_RGBA, GL_FLOAT
        elif internalType == GL_RGBA8 or internalType == GL_RGBA:
            return GL_RGBA, GL_UNSIGNED_BYTE
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

    def getFBO(self, width:int, height:int, depth:bool=False, ms:bool=False, samples:int=1, colors:Optional[Iterable[int]]=None) -> Tuple[int, int]:
        '''
        Get or create a Frame Buffer Object (FBO) with a depth texture.
        If the FBO already exists and the dimensions match, it will return the existing FBO.
        Args:
            width (int): The width of the FBO.
            height (int): The height of the FBO.
        Returns:
            tuple (int, int): A tuple containing the FBO and the depth texture.
        '''

        colors = tuple(colors or [])
        samples = max(1, int(samples))
        ms = bool(ms and samples > 1)
        if not ms:
            samples = 1

        if (self._fbo is None or
            self._width != width or
            self._height != height or
            self._has_depth != bool(depth) or
            self._is_multisample != ms or
            self._samples != samples or
            self._colors != colors):
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
        self._depth_texture = depth_texture
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
        self._depth_texture = depth_texture
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

        if attachment == GL_COLOR_ATTACHMENT0:
            self._color_texture = texID
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

        if attachment == GL_COLOR_ATTACHMENT0:
            self._color_texture = texID
        self._attachments_id.append(texID)

    def _createFBO(self, width:int, height:int, depth:bool=False, ms:bool=False, samples:int=1, colors:Iterable[int]=()):

        # print(f"FBOManager: Creating FBO: {width}x{height}")

        if self._fbo is not None:
            # print(f"FBOManager: Cleaning up existing FBO and attachments: {self._fbo}")
            self.cleanUp()


        colors = tuple(colors or [])
        samples = max(1, int(samples))
        ms = bool(ms and samples > 1)
        if not ms:
            samples = 1

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

        if len(colors):
            glDrawBuffers(len(colors), [GL_COLOR_ATTACHMENT0 + i for i in range(len(colors))])
        else:
            glDrawBuffer(GL_NONE)
            glReadBuffer(GL_NONE)


        # check if the framebuffer is complete
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f'FBOManager: FBO creation failed: {status}')
            exit(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._width = width
        self._height = height
        self._has_depth = bool(depth)
        self._is_multisample = ms
        self._samples = samples
        self._colors = colors

    def textureIndexForColorAttachment(self, colorIndex:int=0) -> int:
        '''
        Return the texture index in _attachments_id for a color attachment.
        '''
        if colorIndex < 0 or colorIndex >= len(self._colors):
            raise ValueError(f'FBOManager: Invalid color attachment index: {colorIndex}, max: {len(self._colors)}.')
        return (1 if self._has_depth else 0) + colorIndex

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

        if self._is_multisample:
            raise RuntimeError('FBOManager: Multisample textures must be resolved before sampler2D reading.')

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

        self._depth_texture = None
        self._color_texture = None
        self._geometry_texture = None
        self._width = 0
        self._height = 0
        self._has_depth = False
        self._is_multisample = False
        self._samples = 1
        self._colors = tuple()


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
    cameraSelectedSignal = Signal(dict)

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


        defaultFormat = QSurfaceFormat.defaultFormat()
        defaultSamples = int(defaultFormat.samples())
        requestedSamples = 4 if defaultSamples < 0 else defaultSamples

        GLFormat = self.format()
        formatChanged = False
        if GLFormat.alphaBufferSize() < 8:
            GLFormat.setAlphaBufferSize(8)
            formatChanged = True
        if GLFormat.depthBufferSize() < 24:
            GLFormat.setDepthBufferSize(24)
            formatChanged = True
        if GLFormat.stencilBufferSize() < 8:
            GLFormat.setStencilBufferSize(8)
            formatChanged = True
        if GLFormat.samples() != requestedSamples:
            GLFormat.setSamples(max(0, requestedSamples))
            formatChanged = True
        if formatChanged:
            self.setFormat(GLFormat)

        self._requestedMSAASamples = max(0, requestedSamples)
        self._enableMSAA = self._requestedMSAASamples > 1
        self._msaaSamples = 0
        self._contextMSAASamples = 0
        self._defaultFramebufferSamples = 0
        self._maxMSAASamples = 0
        self._useOffscreenMSAA = False
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

        self._flatShading = 0

        self.glSettings = GLSettingWidget(
            parent=self,
            render_mode_callback=self.setRenderMode,
            flat_shading_callback=self.setFlatShading,
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
            copy_rgba_callback=self.copyRGBAMapToClipboard,
            enable_ssao_callback=self.setEnableSSAO,
            ssao_kernel_size_callback=self.setSSAOKernelSize,
            ssao_strength_callback=self.setSSAOStrength,
            enable_msaa_callback=self.setEnableMSAA,
            msaa_samples_callback=self.setMSAASamples,
        )

        self.glSettingButton = self.glSettings.get_button()
        self.glCameraPerpCombobox = self.glSettings.gl_camera_perp_combobox
        self.glCameraViewCombobox = self.glSettings.gl_camera_view_combobox


        self.FPSTimer = QTimer()
        self.FPSTimer.timeout.connect(self.countFPS)
        self.FPSTimer.setInterval(1000) # 1 second

        self._fps = 0

        self._lastSavePath = ''

        self._cameraMaskEnabled = False
        self._cameraMaskOpacity = 0.7
        self._cameraOutputResolution: Optional[tuple[int, int]] = None
        self._cameraMaskProg = None
        self._cameraMaskProgLocMap = {}
        self._themeColor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self._cameraConfigs = {}
        self.cameraComboBox = getCameraComboBox(self)
        self.cameraComboBox.setPlaceholderText("Select Camera")
        self.cameraComboBox.hide()
        self.cameraComboBox.currentTextChanged.connect(self._onCameraComboBoxChanged)

        self.canvas2d_scale = 1.0
        self.canvas2d_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.canvas2d_enabled = False
        self._directRightDragAfterCanvas2DExit = False

    @staticmethod
    def _normalizeRGBAColor(color) -> np.ndarray:
        if isinstance(color, QColor):
            return np.array([color.redF(), color.greenF(), color.blueF(), color.alphaF()], dtype=np.float32)

        color = np.asarray(color, dtype=np.float32).flatten()
        if color.size < 3:
            raise ValueError('Theme color must have at least 3 channels')
        if color.size == 3:
            color = np.concatenate([color, np.array([1.0], dtype=np.float32)])
        color = color[:4]
        if np.max(color) > 1.0:
            color = color / 255.0
        return np.clip(color, 0.0, 1.0).astype(np.float32)

    def setThemeColor(self, color):
        self._themeColor = self._normalizeRGBAColor(color)
        self._uploadCameraMaskLineColor()
        self.update()

    def _uploadCameraMaskLineColor(self):
        if self._cameraMaskProg is None:
            return

        loc = self._cameraMaskProgLocMap.get('u_lineColor', -1)
        if loc == -1:
            return

        if isinstance(self.context(), QOpenGLContext) and self.context().isValid():
            self.makeCurrent()

        glUseProgram(self._cameraMaskProg)
        glUniform4f(loc, *[float(v) for v in self._themeColor])
        glUseProgram(0)

    def viewMatrixMod(self, viewMatrix):
        return viewMatrix

    def projMatrixMod(self, projMatrix):
        if not self.canvas2d_enabled:
            return projMatrix

        canvasTransform = np.identity(4, dtype=np.float32)
        canvasTransform[0, 3] = self.canvas2d_offset[0] * self.canvas2d_scale
        canvasTransform[1, 3] = self.canvas2d_offset[1] * self.canvas2d_scale
        canvasTransform[0, 0] = self.canvas2d_scale
        canvasTransform[1, 1] = self.canvas2d_scale

        return projMatrix @ canvasTransform.T

    def reset2DCanvas(self):
        self.canvas2d_scale = 1.0
        self.canvas2d_offset = np.array([0.0, 0.0], dtype=np.float32)
        self._directRightDragAfterCanvas2DExit = False
        self._setCanvas2DEnabled(False)

    def _restoreDefaultCameraIntrinsics(self, isAnimated: bool = True):
        if self.camera.projection_mode != GLCamera.projectionMode.perspective:
            self.camera.setProjectionMode(GLCamera.projectionMode.perspective)
            self.glCameraPerpCombobox.setCurrentItem('0')

        self.camera.setFOV(60)
        self.camera.setIntrinsicPixelOffset(0.0, 0.0)
        self._updateCameraIntrinsicPixelOffset()
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        self.camera.updateProjTransform(isAnimated=isAnimated, isEmit=False)

    def _exitCanvas2DModeToDefaultIntrinsics(self):
        self._setCanvas2DEnabled(False)
        self.setCameraComboBoxtoDefault()
        self.setCameraMaskEnabled(False)
        self._restoreDefaultCameraIntrinsics(isAnimated=True)

    def _syncCameraMotionToCurrentState(self):
        self.camera.filterAEV.stable(np.array([
            self.camera.azimuth,
            self.camera.elevation,
            self.camera.viewPortDistance,
        ]))
        self.camera.filterlookatPoint.stable(self.camera.lookatPoint)
        if self.camera.controltype == self.camera.controlType.arcball:
            self.camera.filterRotaion.stable(self.camera.arcboall_quat)
        self.camera.updateTransform(isAnimated=True, isEmit=False)

    def _syncCanvas2DObjectVisibility(self):
        for obj in self._objectList.values():
            if obj.getProp('hideInCanvas2D', False):
                obj.setProp('canvas2dAutoHidden', self.canvas2d_enabled)

    def _setCanvas2DEnabled(self, enabled: bool):
        self.canvas2d_enabled = bool(enabled)
        self._syncCanvas2DObjectVisibility()
        self.update()

    def addCameraConfig(self, name: str, config: dict, callback: bool = True):
        self._cameraConfigs[name] = config

        self.cameraComboBox.blockSignals(True)
        if self.cameraComboBox.findText(name) == -1:
            self.cameraComboBox.addItem(name)
        self.cameraComboBox.blockSignals(False)

        if len(self._cameraConfigs) == 2:
            self.cameraComboBox.setCurrentText(name)
            if callback:
                self._onCameraComboBoxChanged(name)

        if len(self._cameraConfigs) >= 2:
            self.cameraComboBox.show()
            self.cameraComboBox.raise_()

    def setCameraComboBoxtoDefault(self):
        if "Default" in self._cameraConfigs:
            self.cameraComboBox.blockSignals(True)
            self.cameraComboBox.setCurrentText("Default")
            self.cameraComboBox.blockSignals(False)

    def clearCameraConfigs(self):
        self._cameraConfigs.clear()
        self.cameraComboBox.blockSignals(True)
        self.cameraComboBox.clear()
        self.cameraComboBox.blockSignals(False)
        self.addCameraConfig("Default", {}, callback=False)
        self.cameraComboBox.hide()
        self.setCameraMaskEnabled(False)
        self.clearCameraOutputResolution()
        self.reset2DCanvas()

    def _onCameraComboBoxChanged(self, text: str):
        if text in self._cameraConfigs:
            self.reset2DCanvas()
            if len(self._cameraConfigs[text]):
                self.cameraSelectedSignal.emit(self._cameraConfigs[text])
                self._updateCameraIntrinsicPixelOffset()
                self._setCanvas2DEnabled(True)
                # self.infoSignal.emit("Camera Selected", f"Camera configuration '{text}' selected.", "msg")
            if text == "Default":
                self.resetCamera()

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
            # print(f'Setting background color to: {self._bgColor}')
            if isinstance(self.context(), QOpenGLContext) and self.context().isValid():
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
        self._updateCameraIntrinsicPixelOffset()
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        self.update()

    def setCameraMaskEnabled(self, enabled: bool = False):
        self._cameraMaskEnabled = bool(enabled)
        self._updateCameraIntrinsicPixelOffset()
        self.update()

    def setCameraMaskOpacity(self, opacity: float = 0.7):
        self._cameraMaskOpacity = max(0.0, min(1.0, float(opacity)))
        self.update()

    def setCameraOutputResolution(self, width: int, height: int):
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError(f'Camera output resolution must be positive, got {width}x{height}')
        self._cameraOutputResolution = (width, height)
        self._updateCameraIntrinsicPixelOffset()
        self.update()

    def clearCameraOutputResolution(self):
        self._cameraOutputResolution = None
        self._updateCameraIntrinsicPixelOffset()
        self.update()

    def getCameraMaskSettings(self) -> dict:
        return {
            'enabled': self._cameraMaskEnabled,
            'opacity': self._cameraMaskOpacity,
            'resolution': self._cameraOutputResolution,
        }

    def _calcCameraMaskContentPixelRect(self) -> Optional[np.ndarray]:
        if (not self._cameraMaskEnabled or
            self._cameraOutputResolution is None or
            self._rawWindowW <= 0 or
            self._rawWindowH <= 0):
            return None

        tgt_w, tgt_h = self._cameraOutputResolution
        content_w = min(int(tgt_w), int(self._rawWindowW))
        content_h = min(int(tgt_h), int(self._rawWindowH))
        if content_w >= self._rawWindowW and content_h >= self._rawWindowH:
            return None

        left = int(np.floor((self._rawWindowW - content_w) * 0.5))
        bottom = int(np.floor((self._rawWindowH - content_h) * 0.5))
        right = left + content_w
        top = bottom + content_h
        return np.array([left, bottom, right, top], dtype=np.float32)

    def _calcCameraMaskPixelMargin(self) -> tuple[float, float]:
        if (
            (not self._cameraMaskEnabled and not self.canvas2d_enabled) or
            self._cameraOutputResolution is None or
            self._rawWindowW <= 0 or
            self._rawWindowH <= 0):
            return 0.0, 0.0

        tgt_w, tgt_h = self._cameraOutputResolution
        content_w = min(int(tgt_w), int(self._rawWindowW))
        content_h = min(int(tgt_h), int(self._rawWindowH))
        margin_x = max(float(self._rawWindowW - content_w) * 0.5, 0.0)
        margin_y = max(float(self._rawWindowH - content_h) * 0.5, 0.0)
        return margin_x, margin_y

    def _calcCameraMaskImageCropRect(self, image_width: int, image_height: int) -> Optional[tuple[int, int, int, int]]:
        if (
            not self.camera.useCustomIntrinsic or
            not self._cameraMaskEnabled or
            self._cameraOutputResolution is None or
            image_width <= 0 or
            image_height <= 0):
            return None

        tgt_w, tgt_h = self._cameraOutputResolution
        content_w = min(int(tgt_w), int(image_width))
        content_h = min(int(tgt_h), int(image_height))
        margin_x = max(float(image_width - content_w) * 0.5, 0.0)
        margin_y = max(float(image_height - content_h) * 0.5, 0.0)

        if margin_x <= 0.0 and margin_y <= 0.0:
            return None

        left = int(round(margin_x))
        top = int(round(margin_y))
        right = int(round(float(image_width) - margin_x))
        bottom = int(round(float(image_height) - margin_y))
        crop_w = max(right - left, 1)
        crop_h = max(bottom - top, 1)
        return left, top, crop_w, crop_h

    def _updateCameraIntrinsicPixelOffset(self):
        if self.camera.useCustomIntrinsic:
            margin_x, margin_y = self._calcCameraMaskPixelMargin()
            self.camera.setIntrinsicPixelOffset(margin_x, margin_y)
        else:
            self.camera.setIntrinsicPixelOffset(0.0, 0.0)

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
        print("Resetting camera to default view.")
        self.camera.setCamera(azimuth=135, elevation=-55, distance=5, lookatPoint=np.array([0., 0., 0.,]))
        self._updateCameraIntrinsicPixelOffset()
        self.camera.setFOV(60)
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        self.camera.setLockRotate(False)
        self.glCameraViewCombobox.setCurrentItem('6')
        self.setCameraComboBoxtoDefault()
        self.reset2DCanvas()

        if hasattr(self, 'grid'):
            self.grid.setMode(5)
        if hasattr(self, 'smallGrid'):
            self.smallGrid.setMode(5)

        self.setCameraMaskEnabled(False)
        self.clearCameraOutputResolution()

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
            self.grid.setMode(preset)
            self.camera.setProjectionMode(GLCamera.projectionMode.perspective)
            self.glCameraPerpCombobox.setCurrentItem('0')
            self.camera.setLockRotate(False)
            self._updateCameraIntrinsicPixelOffset()
            self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        else:
            self.camera.setViewPreset(preset)
            self.grid.setMode(preset)
            self.smallGrid.setMode(preset)
            self.camera.setProjectionMode(GLCamera.projectionMode.orthographic)
            self.glCameraPerpCombobox.setCurrentItem('1')
            self.camera.setLockRotate(True)
            self._updateCameraIntrinsicPixelOffset()
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
            if obj.getProp('hideInCanvas2D', False):
                obj.setProp('canvas2dAutoHidden', self.canvas2d_enabled)
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

    def setFlatShading(self, enable:bool=False):
        '''
        Set Flat Shading
        Args:
            enable (bool): Whether to enable flat shading.
        '''
        self._flatShading = 1 if enable else 0
        self.update()

    def buildShader(self, vshader_path:str, fshader_path:str, gshader_path:Optional[str]=None, manualVersion:str='420 core', validate=True) -> int:
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
                program = shaders.compileProgram(vshader, gshader, fshader, validate=validate)
            else:
                program = shaders.compileProgram(vshader, fshader, validate=validate)
            return program

        except Exception as e:
            print(f"Error compiling/linking shaders: {vshader_path} and {fshader_path} \n reason: \n {e}")
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

    def _configureMSAA(self, log:bool=False):
        requestedSamples = self._normalizeMSAASamples(self._requestedMSAASamples) if self._enableMSAA else 0

        if requestedSamples <= 1:
            self._msaaSamples = 0
            self._useOffscreenMSAA = False
        elif self._defaultFramebufferSamples > 1:
            self._msaaSamples = self._defaultFramebufferSamples
            self._useOffscreenMSAA = False
        else:
            self._msaaSamples = requestedSamples
            self._useOffscreenMSAA = True

        if isinstance(self.context(), QOpenGLContext) and self.context().isValid():
            if self._msaaSamples > 1:
                glEnable(GL_MULTISAMPLE)
            else:
                glDisable(GL_MULTISAMPLE)

        if log:
            msaaMode = 'offscreen' if self._useOffscreenMSAA else 'default-fbo'
            print(
                f'OpenGL MSAA samples: requested={self._requestedMSAASamples}, '
                f'enabled={self._enableMSAA}, '
                f'context={self._contextMSAASamples}, '
                f'defaultFBO={self._defaultFramebufferSamples}, '
                f'effective={self._msaaSamples}, mode={msaaMode}, max={self._maxMSAASamples}'
            )

    def _normalizeMSAASamples(self, samples:int) -> int:
        samples = int(samples)
        if samples <= 1:
            return 0

        if self._maxMSAASamples > 0:
            return min(samples, self._maxMSAASamples)
        return samples

    def setEnableMSAA(self, enable:bool=True):
        self._enableMSAA = bool(enable)
        if isinstance(self.context(), QOpenGLContext) and self.context().isValid():
            self.makeCurrent()
            self._configureMSAA(log=True)
        self.update()

    def setMSAASamples(self, samples:int=4):
        self._requestedMSAASamples = max(0, int(samples))
        if isinstance(self.context(), QOpenGLContext) and self.context().isValid():
            self.makeCurrent()
            self._configureMSAA(log=True)
        self.update()

    def setLights(self, program, locmap, lights:Optional[list[PointLight]]=None):
        """
        Set the point lights for the scene.
        Args:
            lights (list[PointLight]): The list of point lights to set.
        Returns:
            None
        """
        lights = lights if lights is not None else self.defaultLights
        numLights = min(len(lights), 5)

        # if hasattr(self, 'SSAOLightProg') and self.SSAOLightProg is not None:
        if program is not None:
            self.makeCurrent()

            glUseProgram(program)

            for i in range(numLights):
                glUniform3f(locmap[f'u_Lights[{i}].position'], *lights[i].position)
                glUniform3f(locmap[f'u_Lights[{i}].color'],    *lights[i].color)

            glUniform1i(locmap['u_NumLights'], numLights)
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

            maxSampleValues = [max(0, int(glGetIntegerv(GL_MAX_SAMPLES)))]
            for pname in (GL_MAX_COLOR_TEXTURE_SAMPLES, GL_MAX_DEPTH_TEXTURE_SAMPLES):
                try:
                    maxSampleValues.append(max(0, int(glGetIntegerv(pname))))
                except Exception:
                    pass
            self._maxMSAASamples = min([v for v in maxSampleValues if v > 0], default=0)
            self._contextMSAASamples = max(0, int(self.context().format().samples()))
            glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
            self._defaultFramebufferSamples = max(0, int(glGetIntegerv(GL_SAMPLES)))
            self._configureMSAA(log=True)

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

            self.SSAOGeoProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_geo_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_geo_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )
            self.SSAOCoreProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_core_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_core_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )
            self.SSAOBlurProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_blur_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_blur_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )
            self.SSAOLightProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )

            self.SSAOLightLineProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_line_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_fs.glsl',
                gshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_line_gs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )

            self.OITAccumProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/oit_accum_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )

            self.OITAccumLineProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_line_vs.glsl',
                gshader_path=f'./glw/shaders/{self.shaderFolder}/ssao_light_line_gs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/oit_accum_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )

            self.OITCompositeProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/oit_composite_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/oit_composite_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )

            self.ScreenCopyProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/oit_composite_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/screen_copy_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )

            self.textProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/text_vs.glsl',
                gshader_path=f'./glw/shaders/{self.shaderFolder}/text_gs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/text_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
            )

            self._cameraMaskProg = self.buildShader(
                vshader_path=f'./glw/shaders/{self.shaderFolder}/camera_mask_vs.glsl',
                fshader_path=f'./glw/shaders/{self.shaderFolder}/camera_mask_fs.glsl',
                manualVersion=shaderVersion,
                validate=not sys.platform == 'darwin'
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
                                        'u_pointSize','u_lineWidth','u_FlatShading'
            ]

            self.textProgUniformList = ['u_mvpMatrix', 'u_AlbedoTexture', 'u_screenSize', 'u_bearingAndSize', 'u_advance', 'u_fontSize', 'u_textColor']

            print('Shaders compiled successfully.')

            self.SSAOGeoProgLocMap = self._cacheShaderLocMap(self.SSAOGeoProg, self.geoProgAttribList, self.geoProgUniformList)
            self.SSAOLightProgLocMap = self._cacheShaderLocMap(self.SSAOLightProg, self.lightProgAttribList, self.lightProgUniformList)

            self.SSAOCoreProgLocMap = self._cacheShaderLocMap(self.SSAOCoreProg, self.coreBlurProgAttribList, self.coreProgUniformList)
            self.SSAOBlurProgLocMap = self._cacheShaderLocMap(self.SSAOBlurProg, self.coreBlurProgAttribList, self.blurProgUniformList)

            self.SSAOLightLineProgLocMap = self._cacheShaderLocMap(self.SSAOLightLineProg, self.lightProgAttribList, self.lightProgUniformList)

            self.OITAccumProgLocMap = self._cacheShaderLocMap(self.OITAccumProg, self.lightProgAttribList, self.lightProgUniformList)
            self.OITAccumLineProgLocMap = self._cacheShaderLocMap(self.OITAccumLineProg, self.lightProgAttribList, self.lightProgUniformList)
            self.OITCompositeProgLocMap = self._cacheShaderLocMap(self.OITCompositeProg, [], ['u_AccumTexture', 'u_RevealTexture'])
            self.ScreenCopyProgLocMap = self._cacheShaderLocMap(self.ScreenCopyProg, [], ['u_Texture'])

            self.textProgLocMap = self._cacheShaderLocMap(self.textProg, [], self.textProgUniformList)

            if self._cameraMaskProg is not None:
                self._cameraMaskProgLocMap = self._cacheShaderLocMap(
                    self._cameraMaskProg,
                    ['a_Position'],
                    ['u_contentPixelRect', 'u_maskAlpha', 'u_lineColor']
                )
                self._uploadCameraMaskLineColor()

            self.SSAOGeoFBO = FBOManager()
            self.SSAOCoreFBO = FBOManager()
            self.SSAOBlurFBO = FBOManager()

            self.SceneFBO = FBOManager()
            self.SceneResolveFBO = FBOManager()
            self.OITFBO = FBOManager()
            self.OITResolveFBO = FBOManager()

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

            self.setLights(self.SSAOLightProg, self.SSAOLightProgLocMap)
            self.setLights(self.OITAccumProg, self.OITAccumProgLocMap)
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
            if not isinstance(obj, Label):
                self._setGeoProgMVPMatrix(locMap, obj.transform, viewMatrix, projMatrix)
                obj.render(locMap=locMap)

    def _renderLabels(self, locMap:dict, viewMatrix:np.ndarray, projMatrix:np.ndarray):
        '''
        A helper function to render all labels in the scene.
        Args:
            locMap (dict): The location map for shader variables.
        '''
        for obj in self._objectList.values():
            if isinstance(obj, Label):
                mvpMatrix = projMatrix @ viewMatrix @ obj.transform
                glUniformMatrix4fv(locMap['u_mvpMatrix'], 1, GL_FALSE, mvpMatrix.T, None)
                glUniform2f(locMap['u_screenSize'], float(self._rawWindowW), float(self._rawWindowH))
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
        glUniform1i(locMap['u_FlatShading'], self._flatShading)



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

    def _prepareSceneRenderTarget(self) -> int:
        '''
        Return the framebuffer used by the final scene passes.
        '''
        if self._useOffscreenMSAA and self._msaaSamples > 1:
            self.SceneFBO.getFBO(
                self._rawWindowW,
                self._rawWindowH,
                depth=True,
                ms=True,
                samples=self._msaaSamples,
                colors=[GL_RGBA8]
            )
            return self.SceneFBO._fbo

        return self.defaultFramebufferObject()

    def _resolveSceneRenderTarget(self):
        '''
        Resolve the offscreen multisample scene target into the Qt widget FBO.
        '''
        if not self._useOffscreenMSAA or self.SceneFBO._fbo is None:
            return

        self.SceneResolveFBO.getFBO(
            self._rawWindowW,
            self._rawWindowH,
            depth=False,
            ms=False,
            colors=[GL_RGBA8]
        )

        self._copyBuffer(self.SceneFBO, self.SceneResolveFBO, GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0)

        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glUseProgram(self.ScreenCopyProg)
        self.SceneResolveFBO.bindTextureForReading(
            GL_TEXTURE24,
            self.SceneResolveFBO.textureIndexForColorAttachment(0)
        )
        glUniform1i(self.ScreenCopyProgLocMap['u_Texture'], 24)
        self.quad.render()
        glUseProgram(0)
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):


        self.camera.setAspectRatio(float(self._scaledWindowW) / float(self._scaledWindowH))
        # self._updateCameraIntrinsicPixelOffset()
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        projMatrix = self.camera.updateProjTransform(isEmit=False)
        projMatrix = self.projMatrixMod(projMatrix)
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

        sceneFramebuffer = self._prepareSceneRenderTarget()

        glBindFramebuffer(GL_FRAMEBUFFER, sceneFramebuffer)
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




        # for obj in self._objectList.values():
        #     if not isinstance(obj, Label):
        #         if not isinstance(obj, UnionObject):
        #             if obj.renderType != GL_LINES:
        #                 glUseProgram(self.SSAOLightProg)
        #                 self._setLightProgMVPMatrix(self.SSAOLightProgLocMap, obj.transform, camtrans, projMatrix.T)
        #                 obj.render(locMap=self.SSAOLightProgLocMap)
        #             else:
        #                 glUseProgram(self.SSAOLightLineProg)
        #                 self._setLightProgMVPMatrix(self.SSAOLightLineProgLocMap, obj.transform, camtrans, projMatrix.T)
        #                 obj.render(locMap=self.SSAOLightLineProgLocMap)
        #         else:
        #             for _obj in obj.objs:
        #                 if _obj.renderType == GL_LINES:
        #                     glUseProgram(self.SSAOLightLineProg)
        #                     self._setLightProgMVPMatrix(self.SSAOLightLineProgLocMap, _obj.transform, camtrans, projMatrix.T)
        #                     _obj.render(locMap=self.SSAOLightLineProgLocMap)
        #                 else:
        #                     glUseProgram(self.SSAOLightProg)
        #                     self._setLightProgMVPMatrix(self.SSAOLightProgLocMap, _obj.transform, camtrans, projMatrix.T)
        #                     _obj.render(locMap=self.SSAOLightProgLocMap)


        opaque_render_list = []
        transparent_render_list = []

        for obj in self._objectList.values():
            if isinstance(obj, Label): continue

            sub_objs = []
            if isinstance(obj, UnionObject):
                sub_objs = obj.objs
            else:
                sub_objs = [obj]

            for o in sub_objs:
                is_trans = False
                if hasattr(o, 'color') and o.color is not None:
                    if len(o.color.shape) == 1 and o.color.shape[0] >= 4:
                        if o.color[3] < 0.99:
                            is_trans = True
                    elif len(o.color.shape) == 2 and o.color.shape[1] >= 4:
                        # For per-vertex colors, check if any alpha is transparent
                        if np.min(o.color[:, 3]) < 0.99:
                            is_trans = True

                # Check for texture transparency if color check passed as opaque
                if not is_trans and hasattr(o, 'material') and o.material is not None:
                    try:
                        tex_image = None
                        if hasattr(o.material, 'baseColorTexture'):
                             tex_image = o.material.baseColorTexture
                        elif hasattr(o.material, 'image'):
                             tex_image = o.material.image

                        if tex_image is not None and hasattr(tex_image, 'mode') and tex_image.mode in ('RGBA', 'LA'):
                             is_trans = True
                    except:
                        pass

                if is_trans:
                    transparent_render_list.append(o)
                else:
                    opaque_render_list.append(o)

        for obj in opaque_render_list:
            if not isinstance(obj, Label):
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



        ''' stage 5: OIT Pass '''
        if len(transparent_render_list) > 0:

            oitMSAA = self._msaaSamples > 1
            oitSamples = self._msaaSamples if oitMSAA else 1
            self.OITFBO.getFBO(
                self._rawWindowW,
                self._rawWindowH,
                depth=True,
                ms=oitMSAA,
                samples=oitSamples,
                colors=[GL_RGBA32F, GL_R32F]
            )

            glBindFramebuffer(GL_READ_FRAMEBUFFER, sceneFramebuffer)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.OITFBO._fbo)
            glBlitFramebuffer(0, 0, self._rawWindowW, self._rawWindowH, 0, 0, self._rawWindowW, self._rawWindowH, GL_DEPTH_BUFFER_BIT, GL_NEAREST)

            self.OITFBO.bindForWriting()
            glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])

            glClearBufferfv(GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])
            glClearBufferfv(GL_COLOR, 1, [1.0, 1.0, 1.0, 1.0])

            glDepthMask(GL_FALSE)
            glEnable(GL_BLEND)
            glDisable(GL_CULL_FACE)


            glBlendEquation(GL_FUNC_ADD)
            glBlendFunci(0, GL_ONE, GL_ONE)
            glBlendFunci(1, GL_ZERO, GL_ONE_MINUS_SRC_COLOR)


            glUseProgram(self.OITAccumLineProg)
            glUniform1i(self.OITAccumLineProgLocMap['u_enableAO'], self._enableSSAO)
            glUniform3f(self.OITAccumLineProgLocMap['u_CamPos'], *campos)
            glUniform1i(self.OITAccumLineProgLocMap['u_renderMode'], self._glRenderMode)
            glUniform2f(self.OITAccumLineProgLocMap['u_screenSize'], float(self._rawWindowW), float(self._rawWindowH))

            glUseProgram(self.OITAccumProg)

            if self._enableSSAO:
                self.SSAOBlurFBO.bindTextureForReading(GL_TEXTURE21, 0)
                glUniform1i(self.OITAccumProgLocMap['u_AOMap'], 21)

            glUniform1i(self.OITAccumProgLocMap['u_enableAO'], self._enableSSAO)
            glUniform3f(self.OITAccumProgLocMap['u_CamPos'], *campos)
            glUniform1i(self.OITAccumProgLocMap['u_renderMode'], self._glRenderMode)
            glUniform2f(self.OITAccumProgLocMap['u_screenSize'], float(self._rawWindowW), float(self._rawWindowH))



            for obj in transparent_render_list:
                if not isinstance(obj, UnionObject):
                    if obj.renderType != GL_LINES:
                        glUseProgram(self.OITAccumProg)
                        self._setLightProgMVPMatrix(self.OITAccumProgLocMap, obj.transform, camtrans, projMatrix.T)
                        obj.render(locMap=self.OITAccumProgLocMap)
                    else:
                        glUseProgram(self.OITAccumLineProg)
                        self._setLightProgMVPMatrix(self.OITAccumLineProgLocMap, obj.transform, camtrans, projMatrix.T)
                        obj.render(locMap=self.OITAccumLineProgLocMap)
                else:
                    for _obj in obj.objs:
                        if _obj.renderType == GL_LINES:
                            glUseProgram(self.OITAccumLineProg)
                            self._setLightProgMVPMatrix(self.OITAccumLineProgLocMap, _obj.transform, camtrans, projMatrix.T)
                            _obj.render(locMap=self.OITAccumLineProgLocMap)
                        else:
                            glUseProgram(self.OITAccumProg)
                            self._setLightProgMVPMatrix(self.OITAccumProgLocMap, _obj.transform, camtrans, projMatrix.T)
                            _obj.render(locMap=self.OITAccumProgLocMap)


            glDepthMask(GL_TRUE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            oitCompositeFBO = self.OITFBO
            if oitMSAA:
                self.OITResolveFBO.getFBO(
                    self._rawWindowW,
                    self._rawWindowH,
                    depth=False,
                    ms=False,
                    colors=[GL_RGBA32F, GL_R32F]
                )
                self._copyBuffer(self.OITFBO, self.OITResolveFBO, GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0)
                self._copyBuffer(self.OITFBO, self.OITResolveFBO, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT1)
                oitCompositeFBO = self.OITResolveFBO

            glBindFramebuffer(GL_FRAMEBUFFER, sceneFramebuffer)
            glUseProgram(self.OITCompositeProg)

            oitCompositeFBO.bindTextureForReading(GL_TEXTURE22, oitCompositeFBO.textureIndexForColorAttachment(0))
            glUniform1i(self.OITCompositeProgLocMap['u_AccumTexture'], 22)
            oitCompositeFBO.bindTextureForReading(GL_TEXTURE23, oitCompositeFBO.textureIndexForColorAttachment(1))
            glUniform1i(self.OITCompositeProgLocMap['u_RevealTexture'], 23)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.quad.render()


        # self._renderObjs(locMap=self.SSAOLightProgLocMap)


        # stage Final: Copy framebuffer to screen (default) framebuffer
        # NOTE: remove before flight

        # self._copyBuffer2Screen(self.SSAOBlurFBO)



        glUseProgram(self.textProg)
        self._renderLabels(self.textProgLocMap, viewMatrix=camtrans, projMatrix=projMatrix.T)

        content_rect = self._calcCameraMaskContentPixelRect()
        if self._cameraMaskProg is not None and content_rect is not None and self._cameraMaskOpacity > 0.0:
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glUseProgram(self._cameraMaskProg)
            glUniform4f(
                self._cameraMaskProgLocMap['u_contentPixelRect'],
                float(content_rect[0]),
                float(content_rect[1]),
                float(content_rect[2]),
                float(content_rect[3]),
            )
            glUniform1f(self._cameraMaskProgLocMap['u_maskAlpha'], float(self._cameraMaskOpacity))
            self.quad.render(locMap=self._cameraMaskProgLocMap)

            glEnable(GL_DEPTH_TEST)

        # mvpMatrix = projMatrix.T @ camtrans
        # glUniformMatrix4fv(glGetUniformLocation(self.textProg, 'u_mvpMatrix'), 1, GL_FALSE, mvpMatrix.T, None)
        # glUniform2f(glGetUniformLocation(self.textProg, 'u_screenSize'), float(self._rawWindowW), float(self._rawWindowH))
        # self.testLabel.render(locMap=self.textProgLocMap)


        glDepthMask(GL_TRUE)

        self._resolveSceneRenderTarget()

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

        self.clearCameraConfigs()
        self.reset2DCanvas()
        self.update()


    def resizeGL(self, w: int, h: int) -> None:
        self._scaledWindowW = w
        self._scaledWindowH = h

        self.pixelRatio = self.devicePixelRatioF()

        self._rawWindowW = int(w * self.pixelRatio)
        self._rawWindowH = int(h * self.pixelRatio)

        self._updateCameraIntrinsicPixelOffset()
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)

        # self.statusbar.move(0, h-self.statusbar.height())
        # self.statusbar.resize(w, h)

        self.glSettings.move((self._scaledWindowW - self.glSettingButton.width()) - 20, 15)
        self.cameraComboBox.move((self._scaledWindowW - self.cameraComboBox.width()) - 20, 15 + self.glSettingButton.height() + 15)

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

        # Handle different projection modes
        if self.camera.projection_mode == self.camera.projectionMode.perspective:
            ...

        elif self.camera.projection_mode == self.camera.projectionMode.orthographic:
            # Orthographic projection: no perspective division needed
            camCoord[2] = -1.0

        else:
            raise ValueError(f'Unknown projection mode: {self.camera.projection_mode}')

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
        u = (u - self._scaledWindowW * self.pixelRatio / 2.0) / self.canvas2d_scale - self.canvas2d_offset[0] * self._scaledWindowW * self.pixelRatio / 2.0 + self._scaledWindowW * self.pixelRatio / 2.0
        v = (v - self._scaledWindowH * self.pixelRatio / 2.0) / self.canvas2d_scale + self.canvas2d_offset[1] * self._scaledWindowH * self.pixelRatio / 2.0 + self._scaledWindowH * self.pixelRatio / 2.0

        camCoord = self.camera.rayVector(int(u), int(v), dis=dis)
        p = camCoord[:3] / camCoord[3]
        # print(f'UV to 3D: {u}, {v} -> {p}')
        return p


    def mousePressEvent(self, event:QMouseEvent):

        self._lastPos = event.pos()
        self._directRightDragAfterCanvas2DExit = False
        self.camera.updateTransform(isAnimated=True, isEmit=False)
        self.update()

        mouseCoordinateinViewPortX = int((self._lastPos.x()) * self.pixelRatio )
        mouseCoordinateinViewPortY = int((self._scaledWindowH -  self._lastPos.y()) * self.pixelRatio)
        mouseCoordinateinViewPortRY = int(self._lastPos.y() * self.pixelRatio)

        self.mouseClickPointinUV = np.array([mouseCoordinateinViewPortX, mouseCoordinateinViewPortY])
        linerDepthValue = self.getDepthPoint(mouseCoordinateinViewPortX, mouseCoordinateinViewPortY)[0]
        self.mouseClickPointinWorldCoordinate = self.camera.rayVector(mouseCoordinateinViewPortX, mouseCoordinateinViewPortY, dis=linerDepthValue)

        # if event.buttons() & Qt.RightButton:
            # transform = np.identity(4, dtype=np.float32)
            # transform[:3, 3] = self.mouseClickPointinWorldCoordinate[:3]
            # self.updateObject(ID=np.random.randint(1, 1000), obj=Label(
            #     'clicked', position=self.mouseClickPointinWorldCoordinate[:3]
            # ))

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
        if self.canvas2d_enabled and event.buttons() & Qt.LeftButton:
            if self._scaledWindowW > 0 and self._scaledWindowH > 0:
                self.canvas2d_offset[0] += dx * 2.0 / self._scaledWindowW / self.canvas2d_scale
                self.canvas2d_offset[1] -= dy * 2.0 / self._scaledWindowH / self.canvas2d_scale
            self.setCameraMaskEnabled(False)
            self._lastPos = event.pos()
            self.update()
            # print(f'Canvas 2D offset updated: {self.canvas2d_offset}')
            return
        elif self.canvas2d_enabled and event.buttons() & Qt.RightButton:
            if abs(dx) <= 1 and abs(dy) <= 1:
                self._lastPos = event.pos()
                self.update()
                return
            self._exitCanvas2DModeToDefaultIntrinsics()
            self._directRightDragAfterCanvas2DExit = True
            # print('Canvas 2D mode disabled due to large mouse movement')
            self.infoSignal.emit('Canvas 2D Mode', 'Canvas 2D mode disabled', 'warning')

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
            if self._directRightDragAfterCanvas2DExit:
                self._syncCameraMotionToCurrentState()

        self._lastPos = event.pos()
        self.mouseMoveSignal.emit(self.mouseClickPointinUV, self.mouseClickPointinWorldCoordinate)
        self.update()

    def wheelEvent(self, event:QWheelEvent):

        angle = event.angleDelta()

        if self.canvas2d_enabled:
            if self._scaledWindowW <= 0 or self._scaledWindowH <= 0:
                return

            zoom_factor = 1.1 if angle.y() > 0 else 0.9

            # mouse pos in NDC
            mx = (event.position().x() / self._scaledWindowW) * 2.0 - 1.0
            my = 1.0 - (event.position().y() / self._scaledWindowH) * 2.0

            # Adjust offset to zoom towards mouse cursor
            self.canvas2d_offset[0] += mx / self.canvas2d_scale * (1.0/zoom_factor - 1.0)
            self.canvas2d_offset[1] += my / self.canvas2d_scale * (1.0/zoom_factor - 1.0)

            self.canvas2d_scale *= zoom_factor
            self.setCameraMaskEnabled(False)
            self.update()
            # print(f'Canvas 2D zoom updated: scale={self.canvas2d_scale}, offset={self.canvas2d_offset}')
            return

        self.camera.zoom(angle.y()/200.)
        self._updateCameraIntrinsicPixelOffset()
        self.camera.updateIntr(self._rawWindowH, self._rawWindowW)
        self.update()

    def mouseDoubleClickEvent(self, event:QMouseEvent) -> None:

        super().mouseDoubleClickEvent(event)

        if event.buttons() & Qt.LeftButton:
            self.resetCamera()

        self.update()

    def mouseReleaseEvent(self, event:QMouseEvent):

        self._directRightDragAfterCanvas2DExit = False
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

    def _grabRGBAMapImage(self):
        image = self.grabFramebuffer()
        crop_rect = self._calcCameraMaskImageCropRect(image.width(), image.height())
        if crop_rect is not None:
            image = image.copy(*crop_rect)
        image = image.convertToFormat(QImage.Format_RGBA8888)
        image.setDevicePixelRatio(1.0)
        return image

    @staticmethod
    def _imageToPngData(image):
        png_data = QByteArray()
        buffer = QBuffer(png_data)
        if not buffer.open(QIODevice.WriteOnly):
            raise RuntimeError('Failed to open clipboard PNG buffer')

        if not image.save(buffer, 'PNG'):
            buffer.close()
            raise RuntimeError('Failed to encode RGBA image as PNG')

        buffer.close()
        return png_data

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
                                                      self._lastSavePath if os.path.exists(os.path.dirname(self._lastSavePath)) else './image.png',
                                                      'PNG Files (*.png);;All Files (*)')

            if path:
                self._lastSavePath = path
                image = self._grabRGBAMapImage()
                image.save(path)
                print(f'RGBA image saved to {path}')
                self.infoSignal.emit('RGBA Image Saved', f'RGBA image saved to {path}', 'complete')
            else:
                print('No path specified to save RGBA image.')
                self.infoSignal.emit('Save RGBA Image', 'No path specified to save RGBA image.', 'warning')

        except Exception as e:
            print(f'Error saving RGBA image: {e}')
            self.infoSignal.emit('Save RGBA Image', f'Error saving RGBA image: {e}', 'error')

    def copyRGBAMapToClipboard(self):
        '''
        Copy the RGBA image to the system clipboard.
        '''
        try:
            image = self._grabRGBAMapImage()
            png_data = self._imageToPngData(image)
            mime_data = QMimeData()
            mime_data.setData('image/png', png_data)
            mime_data.setData('application/x-qt-windows-mime;value="PNG"', png_data)
            mime_data.setData('PNG', png_data)
            mime_data.setImageData(image)
            QApplication.clipboard().setMimeData(mime_data)
            print('RGBA image copied to clipboard')
            self.infoSignal.emit(
                'RGBA Image Copied',
                f'RGBA image copied to clipboard ({image.width()}x{image.height()})',
                'complete')
        except Exception as e:
            print(f'Error copying RGBA image to clipboard: {e}')
            self.infoSignal.emit('Copy RGBA Image', f'Error copying RGBA image to clipboard: {e}', 'error')


    def __del__(self, ):
        try:
            self.reset()
        except:
            ...
