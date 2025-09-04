from PySide6.QtCore import Signal, QObject
from PySide6.QtWidgets import QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from typing import Tuple, Optional, Union
from trimesh.visual.material import SimpleMaterial, PBRMaterial
from enum import Enum
import numpy as np


class BaseObject:

    @property
    def renderType(self): ...
    
    @property
    def isShow(self): ...
    
    @property
    def transform(self): ...

    @property
    def size(self): ...
    
    @property
    def mainColors(self): ...

    @mainColors.setter
    def mainColors(self, value): ...

    @transform.setter
    def transform(self, value: np.ndarray): ...
        
    @renderType.setter
    def renderType(self, value): ...
    
    @isShow.setter
    def isShow(self, value: bool): ...
        
    @size.setter
    def size(self, value): ...

    def setRenderType(self, renderType):
        '''
        Set the rendering type for the object.
        Args:
            renderType (ctypes.c_uint): The rendering type to set. Available types are GL_POINTS, GL_LINES, GL_TRIANGLES, etc.
        '''
        ...

    def setTransform(self, transform:np.ndarray):
        '''
        Set the transformation matrix for the object.
        This is a shorthand for setting the 'transform' property.
        Args:
            transform (np.ndarray): The transformation matrix to set. Must be of shape (4, 4).
        '''
        ...

    def setProp(self, key:str, value):
        '''
        Set a property for the object.
        Args:
            key (str): The property key to set.
            value: The value to set for the property.
        '''
        ...

    def setMultiProp(self, props:dict):
        '''
        Set multiple properties for the object.
        Args:
            props (dict): A dictionary of properties to set.
        '''
        ...
    def getProp(self, key:str, default=None):
        '''
        Get a property for the object.
        Args:
            key (str): The property key to get.
            default: The default value to return if the property is not found.
        Returns:
            The value of the property, or the default value if not found.
        '''
        ...

    def getContext(self): ...
    def setContext(self, context): ...
    def setContextfromCurrent(self): ...
    def makeCurrent(self): ...
    def cleanup(self): ...

class UnionObject(BaseObject):
    def __init__(self) -> None: ...
    def add(self, obj:BaseObject): ...
class PointCloud(BaseObject):
    def __init__(self, vertex:np.ndarray, color:Optional[np.ndarray|list|tuple], norm:Optional[np.ndarray]=None, size=3, transform:Optional[np.ndarray]=None) -> None: ...
    
class Arrow(BaseObject):
    def __init__(self, vertex:np.ndarray, indices:np.ndarray, normal:Optional[np.ndarray]=None, color:Optional[np.ndarray|list|tuple]=(0.8, 0.8, 0.8, 1.0), size:int=3, transform:Optional[np.ndarray]=None) -> None: ...
    
class Grid(BaseObject):
    def __init__(self, n:int=51, scale:float=1.0) -> None: ...
    def setMode(self, mode:int): ...

class Axis(BaseObject):
    def __init__(self, length:float=1, transform:Optional[np.ndarray]=None) -> None: ...

class BoundingBox(BaseObject):
    def __init__(self, vertex:np.ndarray, color:Optional[np.ndarray|list|tuple]=(0.8, 0.8, 0.8, 1.0), norm:Optional[np.ndarray]=None, size:int=3, transform:Optional[np.ndarray]=None) -> None: ...

class Lines(BaseObject):
    def __init__(self, vertex:np.ndarray, color:Optional[np.ndarray|list|tuple]=(0.8, 0.8, 0.8, 1.0), norm:Optional[np.ndarray]=None, size:int=3, transform:Optional[np.ndarray]=None) -> None: ...

class Mesh(BaseObject):
    def __init__(self, vertex:np.ndarray, 
                 indices:np.ndarray, 
                 color:Optional[np.ndarray|list|tuple]=(0.8, 0.8, 0.8, 1.0), 
                 norm:Optional[np.ndarray]=None, 
                 texture:Optional[SimpleMaterial|PBRMaterial]=None, 
                 texcoord:Optional[np.ndarray]=None, 
                 faceNorm:Optional[np.ndarray]=False, 
                 transform:Optional[np.ndarray]=None) -> None: ...

class Sphere(BaseObject):
    def __init__(self, size:float=1.0, transform:Optional[np.ndarray]=None) -> None: ...

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
        ...
    

class PointLight:
    def __init__(self, position:np.ndarray, color:np.ndarray, intensity:float=1.0) -> None:
        ...

class GLWidget(QOpenGLWidget):
    # NOTE: these signals should not be used for internal communication
    leftMouseClickSignal = Signal(np.ndarray, np.ndarray)
    rightMouseClickSignal = Signal(np.ndarray, np.ndarray)
    middleMouseClickSignal = Signal(np.ndarray, np.ndarray)
    mouseReleaseSignal = Signal(np.ndarray, np.ndarray)
    mouseMoveSignal = Signal(np.ndarray, np.ndarray)
    
    camera = GLCamera()

    def enableCountFps(self, enable:bool=True):
        ...

    def setBackgroundColor(self, color: Tuple[float, float, float, float]):
        '''
        This method sets the background color of the OpenGL widget.
        no need to use this func for Windows platform
        Args:
            color (tuple): A tuple of 4 floats representing the RGBA color values, each in the range 0-1.
        Returns:
            None
        '''
        ...

    def setCameraControl(self, index:int):
        '''
        Set camera control type
        Args:
            index (int): The index of the camera control type.
             - 0: Arcball
             - 1: Orbit
        '''
        ...
        
    def setCameraPerspMode(self, index:int):
        '''
        Set camera perspective mode
        Args:
            index (int): The index of the camera perspective mode.
             - 0: Perspective
             - 1: Orthographic
        '''
        ...
        
    def setAxisVisibility(self, isVisible:bool=True):
        '''
        Set axis visibility
        Args:
            isVisible (bool): Whether the axis should be visible or not.
        '''
        ...
        
    def setAxisScale(self, scale:float=1.0):
        '''
        Set axis size
        Args:
            scale (float): The scale factor for the axis.
        '''
        ...
    
    def setGridVisibility(self, isVisible:bool=True):
        '''
        Set grid visibility
        Args:
            isVisible (bool): Whether the grid should be visible or not.
        '''
        ...
    
    def resetCamera(self, ):
        ...

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
        ...
            
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
        ...

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
        ...

    def getObjectList(self, ) -> dict[str, BaseObject]:
        '''
        Get the objects in the objectList.
        Returns:
            dict[str, BaseObject]: A dictionary containing the objects in the objectList.
        '''
        ...

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
        ...

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
        ...

    def buildShader(self, vshader_path:str, fshader_path:str, gshader_path:Optional[str]=None) -> int:
        '''
        Compile and link the vertex and fragment shaders.
        Args:
            vshader_src (str): The source code PATH of the vertex shader.
            fshader_src (str): The source code PATH of the fragment shader.
            gshader_src (Optional[str]): The source code PATH of the geometry shader.
        Returns:
            program (int): The OpenGL program ID.
        '''
        ...

    @staticmethod
    def generateSSAOKernel(kernel_size:int=64) -> np.ndarray:
        """
        Generate SSAO kernel samples.
        Args:
            kernel_size (int): Number of kernel samples, default is 64.
        Returns:
            kernel (np.ndarray (kernel_size, 3)): Array of sample vectors with shape (kernel_size, 3).
        """
        ...

    def setEnableSSAO(self, enable=True):
        """
        Enable or disable SSAO (Screen Space Ambient Occlusion).
        Args:
            enable (bool): True to enable SSAO, False to disable.
        Returns:
            None
        """
        ...

    def setSSAOKernelSize(self, size:int):
        """
        Set the SSAO kernel size.
        Args:
            size (int): The new kernel size.
        Returns:
            None
        """
        ...

    def setSSAOStrength(self, strength:float):
        """
        Set the SSAO strength.
        Args:
            strength (float): The new SSAO strength.
        Returns:
            None
        """
        ...

    def setLights(self, lights:Optional[list[PointLight]]=None):
        """
        Set the point lights for the scene.
        Args:
            lights (list[PointLight]): The list of point lights to set.
        Returns:
            None
        """
        ...

    def setAmbientColor(self, color:Optional[tuple]=None):
        """
        Set the ambient color for the scene.
        Args:
            color (tuple): The new ambient color (R, G, B).
        Returns:
            None
        """
        ...

    def reset(self, ):
        '''
        Clean all Object in the scene.
        '''
        ...

    def worldCoordinatetoUV(self, p:np.ndarray) -> tuple[int, int]:
        '''
        Convert world coordinates to UV coordinates.
        Args:
            p (np.ndarray): The 4D point in world coordinates.

        Returns:
            uv (tuple): The UV coordinates.
        '''
        ...

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
        ...
   
    def getDepthMap(self, ) -> np.ndarray:
        '''
        Get the depth map from the framebuffer. Depth map is converted from NDC to linear space.
        Returns:
            linerDepth (np.ndarray): The linear depth map.
        '''
        ...

    def getDepthPoint(self, x:int, y:int) -> np.ndarray:
        '''
        Get the depth value at a specific pixel location.
        Args:
            x (int): The x-coordinate of the pixel.
            y (int): The y-coordinate of the pixel.
        Returns:
            linerDepth (np.ndarray): The linear depth value.
        '''
        ...

    def saveDepthMap(self, path:Optional[str]=None):
        '''
        Save the depth map to a file to the specified path.
        Args:
            path (Optional[str]): The file path to save the depth map.
        '''
        ...

    def saveRGBAMap(self, path:Optional[str]=None):
        '''
        Save the RGBA image to a file to the specified path.
        Args:
            path (Optional[str]): The file path to save the RGBA image.
        '''
        ...



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
        '''
        Add object to current scene
        
        Args:
            data (dict): dictionary of objects to add
                - key (str): object name
                - value (trimesh.parent.Geometry3D, np.ndarray, dict, None): object data
                if value is None, the object named <key> will be removed from the scene
        Example:
        ```
            data = {
                'pointcloud_#FF0000DD_&10': np.random.rand(100, 1000, 3),
                'lines_#00FF00': np.random.rand(100, 3),
                'box_#0000FF': np.random.rand(100, 8, 3),
                'axis': np.eye(4),
                'mesh': trimesh.load('path/to/mesh.obj'),
                'mesh_2': {'vertex': np.random.rand(100, 3), 'face': np.random.randint(0, 100, (200, 3))},
            }
            b3d.addObj(data)
        ```
        '''
        pass
        
    def add(data:dict):
        '''
        This is an alias of addObj()
        Add object to current scene
        '''
        pass
      
    def updateObj(data:dict):
        '''
        Reset objects in current scene
        Args:
            data (dict): dictionary of objects to set, same as addObj()
        '''
        pass
        
    def rmObj(key:str|list[str]):
        '''
        Remove object named <key> from current scene
        Args:
            key (str or list of str): object name(s) to remove
        '''
        pass

    def rm(key:str|list[str]):
        '''
        This is an alias of rmObj()
        '''
        pass
        
    def getWorkspaceObj() -> dict:
        '''
        Get a copy of the current workspace object dictionary
        Returns:
            workspace_obj (dict): A copy of the current workspace object dictionary
        '''
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
