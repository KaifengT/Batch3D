from PySide6.QtCore import Signal, QObject, QThread, QPoint, QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from typing import Tuple, Optional, Union
from enum import Enum
import numpy as np

class GLCamera(QObject):
    """
    GLCamera is a class that represents a 3D camera in OpenGL.
    It handles camera trajectory, projection, and view matrices.
    """
    
    class controlType(Enum):
        """Camera control type enumeration."""
        arcball = 0
        trackball = 1
        
    class projectionMode(Enum):
        """Camera projection mode enumeration."""
        perspective = 0
        orthographic = 1
    
    updateSignal = Signal()
    
    def setCamera(self, azimuth: float = 0, elevation: float = 50, distance: float = 10, lookatPoint: np.ndarray = None) -> np.ndarray:
        """
        Set the camera position and orientation.
        
        Args:
            azimuth (float): Azimuth angle in degrees. Defaults to 0
            elevation (float): Elevation angle in degrees. Defaults to 50
            distance (float): Distance from the look-at point. Defaults to 10
            lookatPoint (np.ndarray): 3D point to look at. Defaults to origin
            
        Returns:
            np.ndarray: Updated camera transformation matrix (4x4)
        """
        pass
    
    def setCameraTransform(self, transform: np.ndarray) -> np.ndarray:
        """
        Set the camera transformation matrix directly.
        
        Args:
            transform (np.ndarray): 4x4 transformation matrix
            
        Returns:
            np.ndarray: Updated camera transformation matrix (4x4)
        """
        pass
    
    def updateIntr(self, window_h: int, window_w: int) -> np.ndarray:
        """
        Update camera intrinsic matrix based on the window size and view angle.
        
        Args:
            window_h (int): Height of the window
            window_w (int): Width of the window
            
        Returns:
            np.ndarray: Camera intrinsic matrix (3x3)
        """
        pass
    
    def rotate(self, start: Union[float, list] = 0, end: Union[float, list] = 0, window_h: int = 0, window_w: int = 0) -> None:
        """
        Rotate the camera based on mouse drag.
        
        Args:
            start (Union[float, list]): Start point of mouse drag or delta value
            end (Union[float, list]): End point of mouse drag or delta value
            window_h (int): Height of the window. Defaults to 0
            window_w (int): Width of the window. Defaults to 0
            
        Returns:
            None
        """
        pass
    
    def zoom(self, ddistance: float = 0) -> None:
        """
        Zoom the camera by changing the distance.
        
        Args:
            ddistance (float): Change in distance. Defaults to 0
            
        Returns:
            None
        """
        pass
    
    def translate(self, x: float = 0, y: float = 0) -> None:
        """
        Translate the camera look-at point.
        
        Args:
            x (float): Translation in X direction. Defaults to 0
            y (float): Translation in Y direction. Defaults to 0
            
        Returns:
            None
        """
        pass
    
    def translateTo(self, x: float = 0, y: float = 0, z: float = 0, isAnimated: bool = False, isEmit: bool = True) -> None:
        """
        Translate the camera look-at point to specific coordinates.
        
        Args:
            x (float): X coordinate. Defaults to 0
            y (float): Y coordinate. Defaults to 0
            z (float): Z coordinate. Defaults to 0
            isAnimated (bool): Whether to use smooth animation. Defaults to False
            isEmit (bool): Whether to emit update signal. Defaults to True
            
        Returns:
            None
        """
        pass
        
    def updateTransform(self, isAnimated: bool = True, isEmit: bool = True) -> np.ndarray:
        """
        Update the camera transformation matrix.
        
        Args:
            isAnimated (bool): Whether to use smooth animation. Defaults to True
            isEmit (bool): Whether to emit update signal. Defaults to True
            
        Returns:
            np.ndarray: Updated camera transformation matrix (4x4)
        """
        pass
    
    def rayVector(self, ViewPortX: float = 0, ViewPortY: float = 0, dis: float = 1) -> np.ndarray:
        """
        Calculate the ray vector in world coordinates from screen pixel coordinates.
        
        Args:
            ViewPortX (float): Screen pixel X coordinates. Defaults to 0
            ViewPortY (float): Screen pixel Y coordinates. Defaults to 0
            dis (float): Distance along the ray direction. Defaults to 1
            
        Returns:
            np.ndarray: Homogeneous coordinates in world space (4D)
        """
        pass
    
    def updateProjTransform(self, isAnimated: bool = True, isEmit: bool = True) -> np.ndarray:
        """
        Update the projection transformation matrix.
        
        Args:
            isAnimated (bool): Whether to use smooth animation. Defaults to True
            isEmit (bool): Whether to emit update signal. Defaults to True
            
        Returns:
            np.ndarray: Updated projection matrix (4x4)
        """
        pass
    
    def setFOV(self, fov: float = 60.0) -> None:
        """
        Set the field of view angle.
        
        Args:
            fov (float): Field of view angle in degrees. Defaults to 60.0
            
        Returns:
            None
        """
        pass
    
    def setNear(self, near: float = 0.1) -> None:
        """
        Set the near clipping plane distance.
        
        Args:
            near (float): Near clipping plane distance. Defaults to 0.1
            
        Returns:
            None
        """
        pass
    
    def setFar(self, far: float = 4000.0) -> None:
        """
        Set the far clipping plane distance.
        
        Args:
            far (float): Far clipping plane distance. Defaults to 4000.0
            
        Returns:
            None
        """
        pass
    
    def setAspectRatio(self, aspect_ratio: float) -> None:
        """
        Set the aspect ratio of the viewport.
        
        Args:
            aspect_ratio (float): Aspect ratio (width/height)
            
        Returns:
            None
        """
        pass
    
    def setProjectionMode(self, mode) -> None:
        """
        Set the projection mode (perspective or orthographic).
        
        Args:
            mode: Projection mode from projectionMode enum
            
        Returns:
            None
            
        Raises:
            ValueError: If mode is not a valid projection mode
        """
        pass
    
    def setViewPreset(self, preset: int = 0) -> None:
        """
        Set camera to a predefined view preset.
        
        Args:
            preset (int): Preset index. Defaults to 0
                0: +X view
                1: -X view  
                2: +Y view
                3: -Y view
                4: +Z view
                5: -Z view
                
        Returns:
            None
        """
        pass



class GLWidget(QOpenGLWidget):
    # NOTE: these signals should not be used for internal communication
    leftMouseClickSignal = Signal(np.ndarray, np.ndarray)
    rightMouseClickSignal = Signal(np.ndarray, np.ndarray)
    middleMouseClickSignal = Signal(np.ndarray, np.ndarray)
    mouseReleaseSignal = Signal(np.ndarray, np.ndarray)
    mouseMoveSignal = Signal(np.ndarray, np.ndarray)



    scaled_window_w = 0
    scaled_window_h = 0
    raw_window_w = 0
    raw_window_h = 0




    objectList = {}


    
    mouseClickPointinWorldCoordinate = np.array([0,0,0,1])
    mouseClickPointinUV = np.array([0, 0])
    
    camera = GLCamera()

    
    isAxisVisable = True
    isGridVisable = True

    glRenderMode = 3
    pointLineSize = 3
    enableSSAO = 1
    SSAOkernelSize = 64
    SSAOStrength = 60.0
    
    
    depthMap = None


    def setBackgroundColor(color: Tuple[float, float, float, float]) -> None:
        """
        This method sets the background color of the OpenGL widget.
        No need to use this func for Windows platform
        
        Args:
            color (Tuple[float, float, float, float]): A tuple of 4 floats representing the RGBA 
                color values, each in the range 0-1.
                
        Returns:
            None
            
        Raises:
            ValueError: If color values are not in the range 0-1
            AssertionError: If color is not a tuple of 4 floats
        """
        pass

    def setCameraControl(index: int) -> None:
        """
        Set the camera control type.
        
        Args:
            index (int): The index of the camera control type
            
        Returns:
            None
        """
        pass
        
    def setCameraPerspMode(index: int) -> None:
        """
        Set the camera perspective mode (perspective or orthographic).
        
        Args:
            index (int): The index of the projection mode
            
        Returns:
            None
        """
        pass
        
    def setAxisVisibility(isVisible: bool = True) -> None:
        """
        Set the visibility of the coordinate axis.
        
        Args:
            isVisible (bool): True to show axis, False to hide. Defaults to True
            
        Returns:
            None
        """
        pass
        
    def setAxisScale(scale: float = 1.0) -> None:
        """
        Set the scale of the coordinate axis.
        
        Args:
            scale (float): The scale factor for the axis. Defaults to 1.0
            
        Returns:
            None
        """
        pass
    
    def setGridVisibility(isVisible: bool = True) -> None:
        """
        Set the visibility of the grid.
        
        Args:
            isVisible (bool): True to show grid, False to hide. Defaults to True
            
        Returns:
            None
        """
        pass
    
    def resetCamera(self) -> None:
        """
        Reset the camera to its default position and orientation.
        Sets azimuth=135, elevation=-55, distance=10, and lookAt point to origin.
        
        Returns:
            None
        """
        pass

    def setCameraViewPreset(preset: int = 0) -> None:
        """
        Setting the camera view preset.
        
        Args:
            preset (int): index from 0-6. Defaults to 0
                0: Front View
                1: Back View
                2: Left View
                3: Right View
                4: Top View
                5: Bottom View
                6: Free View
                
        Returns:
            None
        """
        pass
            
    def setObjectProps(ID: Union[int, str], props: dict) -> None:
        """
        Setting the properties of an object in the objectList.
        
        Args:
            ID (Union[int, str]): The ID of the object in the objectList
            props (dict): A dictionary containing the properties to be updated.
                Available properties include:
                - 'size': Size of the object (float)
                - 'isShow': Visibility of the object (boolean)
                
        Returns:
            None
        """
        pass

    def setObjTransform(ID: Union[int, str] = 1, transform: Optional[np.ndarray] = None) -> None:
        """
        Setting the transformation matrix of an object in the objectList.
        
        Args:
            ID (Union[int, str]): The ID of the object in the objectList. Defaults to 1
            transform (Optional[np.ndarray]): The homogeneous transformation matrix (4x4) to be set.
                If None, the transformation matrix will be set to the identity matrix. Defaults to None
                
        Returns:
            None
        """
        pass

    def updateObject(ID: Union[int, str] = 1, obj: Optional[object] = None) -> None:
        """
        Update the object in the objectList with a new object or remove it if obj is None.
        
        Args:
            ID (Union[int, str]): The ID of the object in the objectList. Defaults to 1
            obj (Optional[object]): The new object to be set. 
                If None, the object which name matches the ID will be removed from the list. Defaults to None
                
        Returns:
            None
        """
        pass

    def setRenderMode(mode: int) -> None:
        """
        Set the rendering mode for the OpenGL widget.
        
        Args:
            mode (int): The rendering mode identifier
            
        Returns:
            None
        """
        pass

    def setEnableSSAO(enable: bool = True) -> None:
        """
        Enable or disable SSAO (Screen Space Ambient Occlusion).
        
        Args:
            enable (bool): True to enable SSAO, False to disable. Defaults to True
            
        Returns:
            None
        """
        pass

    def setSSAOKernelSize(size: int) -> None:
        """
        Set the SSAO kernel size.
        
        Args:
            size (int): The new kernel size
            
        Returns:
            None
        """
        pass

    def setSSAOStrength(strength: float) -> None:
        """
        Set the SSAO strength.
        
        Args:
            strength (float): The new SSAO strength
            
        Returns:
            None
        """
        pass

    def reset(self) -> None:
        """
        Reset the OpenGL widget by clearing all objects and updating the display.
        
        Returns:
            None
        """
        pass

    def resizeGL(w: int, h: int) -> None:
        """
        Handle resizing of the OpenGL widget.
        
        Args:
            w (int): New width of the widget
            h (int): New height of the widget
            
        Returns:
            None
        """
        pass
    
    def getDepthMap(self) -> np.ndarray:
        """
        Get the depth map from the current framebuffer.
        
        Returns:
            np.ndarray: 2D array containing linear depth values
        """
        pass
    
    def getDepthPoint(x: int, y: int) -> np.ndarray:
        """
        Get the depth value at a specific screen coordinate.
        
        Args:
            x (int): X coordinate on screen
            y (int): Y coordinate on screen
            
        Returns:
            np.ndarray: Linear depth value at the specified point
        """
        pass
    
    def saveDepthMap(path: Optional[str] = None) -> None:
        """
        Save the current depth map to a file.
        
        Args:
            path (Optional[str]): File path to save the depth map. If None, opens a file dialog. Defaults to None
            
        Returns:
            None
        """
        pass
            
    def saveRGBAMap(path: Optional[str] = None) -> None:
        """
        Save the current RGBA image to a file.
        
        Args:
            path (Optional[str]): File path to save the image. If None, opens a file dialog. Defaults to None
            
        Returns:
            None
        """
        pass



class b3d(QMainWindow):
    
    # NOTE: this signal should not be used for internal
    workspaceUpdatedSignal = Signal(dict)
    GL = GLWidget()
    
    def resetScriptNamespace():
        # delete objects in script namespace
        pass
        
    def openFolder(path=None):
        pass

    def changeObjectProps():
        pass
            
    def setObjectProps(key, props:dict):
        pass

    def setWorkspaceObj(obj):
        pass
        
    def isSliceable(obj):
        pass

    def slicefromBatch(batch,):
        pass

    def resetSliceFunc():
        pass
        
    def addObj(data:dict):
        pass
        
    def add(data:dict):
        pass
      
    def updateObj(data:dict):
        pass
        
    def rmObj(key:str|list[str]):
        """Remove object from OpenGLWidget"""
        pass

    def rm(key:str|list[str]):
        """Remove object from OpenGLWidget"""
        pass
        
    def getWorkspaceObj() -> dict:
        """Get the current workspace object"""
        pass
        
    def clear():
        pass
                
    def loadObj(fullpath:str|dict, extName='', setWorkspace=True):
        pass
        
    def loadObj_update(fullpath:str|dict, keys:list, extName='', setWorkspace=True):
        pass
        
    def setObjTransform(ID, transform=None):
        """
        Set the transformation matrix for an object in the OpenGL widget.
        
        Parameters:
        - ID: The identifier for the object.
        - transform: A 4x4 transformation matrix.
        """
        pass

    def getFilePathFromList(row:int):
        pass

    def getListLength():
        pass

    def popMessage(title:str='', message:str='', mtype='msg', followMouse=False):
        pass

    def closeEvent(event) -> None:
        pass

    def openRemoteUI():
        pass
        
    def openDetailUI():
        pass
            
    def loadSettings():
        pass

    def saveSettings():
        pass
            
    def showConsole():
        pass
