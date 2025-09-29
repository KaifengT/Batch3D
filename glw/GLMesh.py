# -*- coding: utf-8 -*-
from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import List
from typing import Optional, Union
import PIL.Image as im
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from ctypes import *
from .utils.transformations import invHRT, rotation_matrix, rotationMatrixY, rpy2hRT
from PIL import Image
from trimesh.visual.material import SimpleMaterial, PBRMaterial
from PySide6.QtGui import QOpenGLContext
import freetype

DEFAULT_COLOR3 = np.array([0.8, 0.8, 0.8], dtype=np.float32)
DEFAULT_COLOR4 = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
class colorManager:
    _color =  np.array([    
        # [217, 217, 217], # rgb(217, 217, 217),
        [233, 119, 119], # rgb(233, 119, 119),
        [65 , 157, 129], # rgb(65 , 157, 129),
        [156, 201, 226], # rgb(156, 201, 226),
        [228, 177, 240], # rgb(228, 177, 240),
        [252, 205, 42 ], # rgb(252, 205, 42 ),
        [240, 90 , 126], # rgb(240, 90 , 126),
        [33 , 155, 157], # rgb(33 , 155, 157),
        [114, 191, 120], # rgb(114, 191, 120),
        [199, 91 , 122], # rgb(199, 91 , 122),
        [129, 180, 227], # rgb(129, 180, 227),
        [115, 154, 228], # rgb(115, 154, 228),
        [119, 228, 200], # rgb(119, 228, 200),
        [243, 231, 155], # rgb(243, 231, 155),
        [248, 160, 126], # rgb(248, 160, 126),
        [206, 102, 147], # rgb(206, 102, 147),

        ]) / 255.
        
    def __init__(self):
        self.current_index = 0

    def get_next_color(self):
        color = self._color[self.current_index]
        self.current_index = (self.current_index + 1) % len(self._color)
        return color
    
    def reset(self):
        self.current_index = 0
        
    @staticmethod
    def u8color(color: np.ndarray) -> np.ndarray:
        """Convert float color to uint8 color."""
        return (color * 255).astype(np.uint8)
    
    @staticmethod
    def float_color(color: np.ndarray) -> np.ndarray:
        """Convert uint8 color to float color."""
        return color.astype(np.float32) / 255.0
    
    @staticmethod
    def extract_dominant_colors(colors, n_colors=2, **kwargs):
        return colorManager._extract_histogram(colors, n_colors, **kwargs)

    
    @staticmethod
    def _extract_histogram(colors, n_colors, bins=16, **kwargs):

        if np.max(colors) > 1:
            colors = colors / 255.0
        
        colors_reshaped = colors[..., :3]
        colors_reshaped = colors_reshaped.reshape(-1, 3)

        valid_mask = ~np.any(np.isnan(colors_reshaped), axis=1)
        colors_valid = colors_reshaped[valid_mask]
        
        if len(colors_valid) == 0:
            return np.zeros((n_colors, 3)), np.zeros(n_colors)
        
        quantized = np.floor(colors_valid * bins).astype(int)
        quantized = np.clip(quantized, 0, bins - 1)
        
        indices = (quantized[:, 0] * bins * bins + 
                  quantized[:, 1] * bins + 
                  quantized[:, 2])
        
        unique_indices, counts = np.unique(indices, return_counts=True)
        
        top_n = min(n_colors, len(unique_indices))
        top_indices = np.argsort(counts)[-top_n:][::-1]
        
        dominant_colors = []
        for i in top_indices:
            idx = unique_indices[i]
            r = (idx // (bins * bins)) / bins + 1/(2*bins)
            g = ((idx % (bins * bins)) // bins) / bins + 1/(2*bins)
            b = (idx % bins) / bins + 1/(2*bins)
            dominant_colors.append([r, g, b])
        
        dominant_colors = np.array(dominant_colors)
        percentages = counts[top_indices] / len(colors_valid)
        
        return dominant_colors, percentages
    
    @staticmethod
    def get_color_from_tex(tex: Image.Image, texcoord: np.ndarray) -> np.ndarray:
        """
        Get color from texture image based on texture coordinates.
        
        Args:
            tex (PIL.Image): Texture image.
            texcoord (np.ndarray): Texture coordinates in range [0, 1].
        
        Returns:
            np.ndarray: Color at the specified texture coordinates.
        """
        if isinstance(tex, im.Image):
            image = np.array(tex)
        
        tex_h, tex_w = image.shape[:2]
        texcoord_int = (texcoord * np.array([tex_w, tex_h])).astype(np.int32)
        indices_clipped = np.clip(texcoord_int, [0, 0], [tex_w-1, tex_h-1])
        return image[indices_clipped[:, 1], indices_clipped[:, 0]] / 255.0

    @staticmethod
    def srgb2Linear(arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, 0.0, 1.0)
        low = arr <= 0.04045
        out = np.empty_like(arr)
        out[low] = arr[low] / 12.92
        out[~low] = ((arr[~low] + 0.055) / 1.055) ** 2.4
        return out

    @staticmethod
    def linear2Srgb(arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, 0.0, 1.0)
        low = arr <= 0.0031308
        out = np.empty_like(arr)
        out[low] = arr[low] * 12.92
        out[~low] = 1.055 * (arr[~low] ** (1/2.4)) - 0.055
        return out



class VAOManager:
    '''
    Manages the Vertex Array Object (VAO) and its associated Vertex Buffer Objects (VBOs) and Element Buffer Object (EBO).
    '''
    def __init__(self):
        '''
        Initialize the VAOManager. You should call createVAO() after initializing an OpenGL context.
        '''
        self._vaoid = 0
        self._vboids = {}
        self._eboid = 0
        self._eboLen = 0
        self._len = 0
        
        self._context = None

    def createVAO(self) -> int:
        '''
        Create a Vertex Array Object (VAO). This function generates a new VAO if one does not already exist, 
        and this function should be called after initializing an OpenGL context.
        '''
        if self._vaoid != 0:
            return self._vaoid
        self._vaoid = glGenVertexArrays(1)
        return self._vaoid
    
    def bind(self, ):
        '''
        Bind the Vertex Array Object (VAO) for the current context.
        '''
        if self._vaoid != 0:
            glBindVertexArray(self._vaoid)
        else:
            print('VAO not created!')
            
    def unbind(self, ):
        '''
        Unbind the Vertex Array Object (VAO) for the current context.
        '''
        glBindVertexArray(0)

    def cleanup(self, ):
        '''
        Cleanup the VAOManager by deleting the VAO, VBOs, and EBO.
        '''
        for vbo in self._vboids.values():
            if vbo != 0:
                glDeleteBuffers(1, [vbo])
                # print(f'VBO {vbo} deleted!')

        if self._eboid != 0:
            glDeleteBuffers(1, [self._eboid])
            # print(f'EBO {self._eboid} deleted!')
            self._eboid = 0
            self._eboLen = 0
            
        if self._vaoid != 0:
            glDeleteVertexArrays(1, [self._vaoid])
            # print(f'VAO {self._vaoid} deleted!')
            self._vaoid = 0    

        self._len = 0
        self._vboids.clear()
        

    @staticmethod
    def _createVBO(data:np.ndarray):
        data = np.float32(data)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return vbo
    
    @staticmethod
    def _createEBO(data:np.ndarray):
        data = np.uint32(data.ravel())
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        return ebo, len(data)

    def bindVertexBuffer(self, index:int, data:np.ndarray, size:int, type:ctypes.c_uint=GL_FLOAT, nbytes:int=4):
        '''
        Bind the vertex buffer object (VBO) for the current VAO.
        This function creates a new VBO and binds it to the specified vertex attribute index.
        Args:
            index (int): The index of the vertex attribute.
            data (np.ndarray): The vertex data to bind.
            size (int): The number of components per vertex attribute. If data is position, size should be 3.
            type (ctypes.c_uint, optional): The data type of each component. Defaults to GL_FLOAT.
            nbytes (int, optional): The size of each component in bytes. Defaults to 4.
        '''
        if self._vboids.get(index, 0) != 0:
            glDeleteBuffers(1, [self._vboids[index]])
            self._vboids.pop(index)

        if self._vaoid != 0:
            
            vertexlen = len(data.flatten()) // size
            if self._len != 0 and vertexlen != self._len:
                raise ValueError(f'Warning: Vertex length mismatch! Expected {self._len}, got {vertexlen}')
            self._len = vertexlen

            vbo = self._createVBO(data)
            self._vboids.update({index: vbo})
            
            glBindVertexArray(self._vaoid)
            glEnableVertexAttribArray(index)
            
            glBindBuffer(GL_ARRAY_BUFFER, vbo)

            glVertexAttribPointer(index, size, type, GL_FALSE, size * nbytes, ctypes.c_void_p(0))

            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        else:
            print('bindVertexBuffer: VAO not created!')


    def bindElementBuffer(self, data:np.ndarray):
        '''
        Bind the element buffer object (EBO) for current VAO.
        This function creates a new EBO and binds it to the current VAO.
        Args:
            data (np.ndarray): The index data for the EBO. Type should be uint32.
        '''
        if self._eboid != 0:
            glDeleteBuffers(1, [self._eboid])
            self._eboid = 0
            self._eboLen = 0
        
        
        if self._vaoid != 0:
            if self._eboid == 0:

                # assert data.dtype == np.uint32, 'EBO data must be of type uint32'
                data = data.astype(np.uint32)

                self._eboid, self._eboLen = self._createEBO(data)
                
                glBindVertexArray(self._vaoid)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._eboid)
                glBindVertexArray(0)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            else:
                print('bindElementBuffer: EBO already exists!')
        else:
            print('bindElementBuffer: VAO not created!')

    def getEBOLen(self, ):
        '''
        Get the length of the element buffer.
        Returns:
            int: The number of elements in the buffer.
        '''
        return self._eboLen

    def setVertexLen(self, length:int):
        '''
        Set the length of the vertex buffer.
        Args:
            length (int): The number of vertices in the buffer.
        '''
        self._len = length
        
    def getVertexLen(self, ):
        '''
        Get the length of the vertex buffer.
        Returns:
            int: The number of vertices in the buffer.
        '''
        return self._len

    def getVAO(self):
        '''
        Get the VAO ID for the current mesh.
        Returns:
            int: The VAO ID.
        '''
        return self._vaoid

    def __del__(self, ):
        try:
            self.cleanup()
        except:
            ...

class BaseObject:
    def __init__(self):
        self.vao = VAOManager()
        self.__renderType = GL_POINTS
        

        
        self.__props = {
            'isShow':True,
            'size':3,
            'transform':np.eye(4, dtype=np.float32)
        }


        self.__mainColors = DEFAULT_COLOR3[None]

    @property
    def renderType(self):
        return self.__renderType
    
    @property
    def isShow(self):
        return self.__props.get('isShow', True)
    
    @property
    def transform(self):
        return self.__props.get('transform', np.eye(4, dtype=np.float32))

    @property
    def size(self):
        return self.__props.get('size', 3)
    
    @property
    def mainColors(self):
        return self.__mainColors

    @mainColors.setter
    def mainColors(self, value):
        self.__mainColors = value

    @transform.setter
    def transform(self, value: np.ndarray):
        self.setTransform(value)
        
    @renderType.setter
    def renderType(self, value: ctypes.c_uint):
        self.setRenderType(value)
        
    @isShow.setter
    def isShow(self, value: bool):
        self.setProp('isShow', value)
        
    @size.setter
    def size(self, value):
        self.setProp('size', value)

    def setRenderType(self, renderType:ctypes.c_uint=GL_POINTS):
        '''
        Set the rendering type for the object.
        Args:
            renderType (ctypes.c_uint): The rendering type to set. Available types are GL_POINTS, GL_LINES, GL_TRIANGLES, etc.
        '''
        assert renderType in (GL_POINTS, GL_LINES, GL_TRIANGLES, GL_QUADS, GL_LINE_STRIP, GL_LINE_LOOP, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN), 'Invalid render type'
        self.__renderType = renderType

    def setTransform(self, transform:np.ndarray):
        '''
        Set the transformation matrix for the object.
        This is a shorthand for setting the 'transform' property.
        Args:
            transform (np.ndarray): The transformation matrix to set. Must be of shape (4, 4).
        '''
        if transform is None:
            return
        assert isinstance(transform, np.ndarray), 'transform must be a numpy array'
        assert transform.shape == (4, 4), 'transform shape error'
        self.setProp('transform', transform)

    def setProp(self, key:str, value):
        '''
        Set a property for the object.
        Args:
            key (str): The property key to set.
            value: The value to set for the property.
        '''
        self.__props.update({key:value})

    def setMultiProp(self, props:dict):
        '''
        Set multiple properties for the object.
        Args:
            props (dict): A dictionary of properties to set.
        '''
        self.__props.update(props)

    def getProp(self, key:str, default=None):
        '''
        Get a property for the object.
        Args:
            key (str): The property key to get.
            default: The default value to return if the property is not found.
        Returns:
            The value of the property, or the default value if not found.
        '''
        return self.__props.get(key, default)

    def getContext(self) -> QOpenGLContext:
        return self._context

    def setContext(self, context:QOpenGLContext):
        self._context = context

    def setContextfromCurrent(self):
        self._context = QOpenGLContext.currentContext()

    def makeCurrent(self):
        try:
            if self._context is not None and self._context.isValid():
                self._context.makeCurrent(self._context.surface())
                return True
            else:
                return False
        except:
            return False

    def cleanup(self):

        if self._context is not None and isinstance(self._context, QOpenGLContext) and self._context != QOpenGLContext.currentContext():
            if self._context.isValid():
                self._context.makeCurrent(self._context.surface())
            else:
                print('Invalid context')

        if hasattr(self, 'materialParameter') and isinstance(self.materialParameter, Mapping):

            datas = list(self.materialParameter.values())
            toDelete = []
            for data in datas:
                if data['type'] == 'texture':
                    toDelete.append(data['data'])
            if len(toDelete):
                glDeleteTextures(len(toDelete), toDelete)
                # print(f'Deleted textures: {toDelete}')
            del self.materialParameter
        
        
        self.vao.cleanup()
        
    def __del__(self, ):
        try:
            self.cleanup()
        except:
            ...

    @staticmethod
    def createTexture2d(texture_file:Image.Image):
        
        internalformatMap = {
            GL_RGB: GL_RGB8,
            GL_RGBA: GL_RGBA8,
            GL_RED: GL_R8,
            GL_RG: GL_RG8,
        }
        
        formatMap = {
            4: GL_RGBA,
            3: GL_RGB,
            2: GL_RG,
            1: GL_RED,
            0: GL_RED,
        }
        
        typeMap = {
            np.uint8: GL_UNSIGNED_BYTE,
            np.uint16: GL_UNSIGNED_SHORT,
            np.uint32: GL_UNSIGNED_INT,
            np.int8: GL_BYTE,
            np.int16: GL_SHORT,
            np.int32: GL_INT,
            np.float32: GL_FLOAT,
            np.float16: GL_HALF_FLOAT,
        }
        
        im = np.array(texture_file)
        im_h, im_w = im.shape[:2]
        # im_mode = GL_LUMINANCE if im.ndim == 2 else (GL_RGB, GL_RGBA)[im.shape[-1]-3]
        
        if im.ndim == 2:
            im = im[..., None]
        if im.ndim == 3 and im.shape[-1] == 1:
            im = np.concatenate([im, im, im], axis=-1)

        _type = typeMap.get(im.dtype.type, GL_UNSIGNED_BYTE)
        _dim = 1 if im.ndim < 3 else im.shape[-1]
        _format = formatMap[_dim]
        _iformat = internalformatMap[_format]

        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)

        if (im.size/im_h)%4 == 0:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        else:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
        glTexImage2D(GL_TEXTURE_2D, 0, _iformat, im_w, im_h, 0, _format, _type, im)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        return tid

    @staticmethod
    def buildfaceNormal(vertex, indices):
        primitive = vertex[indices]
        a = primitive[::3]
        b = primitive[1::3]
        c = primitive[2::3]
        face_norm = np.repeat(np.cross(b-a, c-a), 3, axis=0)
        
        ind_rev = np.array(range(vertex.shape[0]))
        ind_rev = ind_rev[indices]
        
        norm = np.zeros_like(vertex)
        norm[ind_rev] = face_norm
        
        return norm
    
    @staticmethod
    def faceNormal2VertexNormal(faceNormal, vertex, indices):
        '''
        vertex (v, 3)
        faceNormal (n, 3)
        indices (n, )
        '''
        
        ind_rev = np.array(range(vertex.shape[0]))
        ind_rev = ind_rev[indices]
        
        norm = np.zeros_like(vertex)
        norm[ind_rev] = faceNormal
        
        return norm

    @staticmethod
    def _decodeStr2RGB(hexcolor:str) -> np.ndarray:
        try:
            if len(hexcolor) >= 6:
                hexcolor = hexcolor + 'FF'

            if len(hexcolor) >= 8:
                return np.array([int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4, 6)], dtype=np.float32)
            else:
                return DEFAULT_COLOR4
        except:
            return DEFAULT_COLOR4
 


    @staticmethod
    def checkData(shape, data:np.ndarray, lastdim:float=4, fill:float=1.0):
        '''
        Check and reshape the input data to match the desired shape and dimensions.
        Args:
            shape (tuple): The desired shape of the output data.
            data (np.ndarray): The input data to check and reshape.
            lastdim (float): The last dimension size of the output data.
            fill (float): The fill value for any missing elements.
        Returns:
            checkedData (np.ndarray): The reshaped output data.
        '''
        data = data.astype(np.float32)
        if data.shape[:-1] == shape[:-1]:
            if data.shape[-1] == lastdim:
                return data
            else:
                _data = np.ones((*data.shape[:-1], lastdim), dtype=np.float32) * fill
                _data[..., :min(data.shape[-1], lastdim)] = data[..., :min(data.shape[-1], lastdim)]
                return _data
            
        else:
            if data.ndim > 1:
                data = data.ravel()[:lastdim]
            
            one = np.ones((*shape[:-1], lastdim), dtype=np.float32)
            
            if data.shape[-1] == lastdim:
                _data = data
            else:
                _data = np.ones((lastdim), dtype=np.float32) * fill
                _data[:min(data.shape[-1], lastdim)] = data[:min(data.shape[-1], lastdim)]
                
            return one * _data



    @staticmethod
    def checkColor(shape, color:Optional[np.ndarray | list | tuple | str]=None):
        if color is None:
            color = DEFAULT_COLOR4
        
        elif isinstance(color, str):
            color = BaseObject._decodeStr2RGB(color)
            
        elif isinstance(color, (list, tuple)):
            color = np.array(color, dtype=np.float32)
            
        assert isinstance(color, np.ndarray), 'color must be a numpy array, list, tuple, hex string or None'
    
        return BaseObject.checkData(shape, color, lastdim=4, fill=1.0)

    @staticmethod
    def _parseMaterial(material:SimpleMaterial|PBRMaterial, texcoord=None):
        
        '''
        trimesh.visual.material.PBRMaterial:
            - name(str)
        
            - baseColorFactor(np.ndarray(4, uint8))
            - baseColorTexture
            
            - emissiveFactor(float)
            - emissiveTexture
            
            - metallicFactor(float)
            - metallicRoughnessTexture
            

            - occlusionTexture
            - normalTexture

            - roughnessFactor(float)
            
            - main_color(np.ndarray(4, uint8))
            
        trimesh.visual.material.SimpleMaterial
            - name(str)
            - ambient(np.ndarray(4, uint8))
            - diffuse(np.ndarray(4, uint8))
            - specular(np.ndarray(4, uint8))
            - shininess(float)
            - glossiness(float)
            - image(PIL.Image)
            - main_color(np.ndarray(4, uint8))

        '''
        def _safeGetArray(color:np.ndarray):
            if not isinstance(color, np.ndarray):
                return None
            color = color.astype(np.float32).flatten() / 255.
            if len(color) == 3:
                color = np.concatenate((color, [1.]), axis=0)
            if len(color) < 4:
                return None
            
            color = color[:4]
            linear = colorManager.linear2Srgb(color)
            # linear = color
            return linear


        miscParamter = {
            'vertexColor': {'data': _safeGetArray(np.array([*DEFAULT_COLOR4])), 'type': 'list'},
        }
        materialParameter = BaseObject.getDefaultMaterialParameter()

        if texcoord is None:
            # find simple color
            if isinstance(material, SimpleMaterial):
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.diffuse), 'type': 'list'}})

            elif isinstance(material, PBRMaterial) and hasattr(material, 'baseColorFactor'):
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.baseColorFactor), 'type': 'list'}})
                
                if isinstance(material.metallicFactor, float):
                    materialParameter.update({'u_Metallic': {'data': material.metallicFactor, 'type': 'float'}})
                if isinstance(material.roughnessFactor, float):
                    materialParameter.update({'u_Roughness': {'data': material.roughnessFactor, 'type': 'float'}})

            return materialParameter, miscParamter

        if isinstance(material, SimpleMaterial):

            if isinstance(material.image, im.Image):
                tid = BaseObject.createTexture2d(material.image)

                texcolor = colorManager.get_color_from_tex(material.image, texcoord)
                main_colors, per = colorManager.extract_dominant_colors(texcolor, n_colors=3)

                materialParameter.update({'u_AlbedoTexture': {'data':tid, 'type':'texture'},
                                         'u_EnableAlbedoTexture': {'data': 1, 'type': 'int'}})
                miscParamter.update({'mainColors':{'data':main_colors, 'type':'list'}})

            else:
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.diffuse), 'type': 'list'}})

            return materialParameter, miscParamter

        elif isinstance(material, PBRMaterial):
            
            if isinstance(material.baseColorTexture, im.Image):
                
                tid = BaseObject.createTexture2d(material.baseColorTexture)
                
                texcolor = colorManager.get_color_from_tex(material.baseColorTexture, texcoord)
                main_colors, per = colorManager.extract_dominant_colors(texcolor, n_colors=3)

                materialParameter.update({'u_AlbedoTexture': {'data': tid, 'type': 'texture'},\
                                        'u_EnableAlbedoTexture': {'data': 1, 'type': 'int'}})
                miscParamter.update({'mainColors': {'data': main_colors, 'type': 'list'}})

                
            else:
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.baseColorFactor), 'type': 'list'}})
                
                
            if isinstance(material.metallicRoughnessTexture, im.Image):
                tid = BaseObject.createTexture2d(material.metallicRoughnessTexture)
                materialParameter.update({'u_MetallicRoughnessTexture': {'data': tid, 'type': 'texture'},
                                         'u_EnableMetallicRoughnessTexture': {'data': 1, 'type': 'int'}})
            else:
                
                if isinstance(material.metallicFactor, float):
                    materialParameter.update({'u_Metallic': {'data': material.metallicFactor, 'type': 'float'}})
                if isinstance(material.roughnessFactor, float):
                    materialParameter.update({'u_Roughness': {'data': material.roughnessFactor, 'type': 'float'}})

            return materialParameter, miscParamter
        else:
            raise ValueError('Unknown material type')

    @staticmethod
    def getDefaultMaterialParameter() -> dict:
        return {'u_Metallic': {'data': 0.0, 'type': 'float'},
                'u_Roughness': {'data': 0.0, 'type': 'float'},
                'u_EnableAlbedoTexture': {'data': 0, 'type': 'int'},
                'u_EnableMetallicRoughnessTexture': {'data': 0, 'type': 'int'}}


    def load(self):
        ...

    def render(self, locMap:dict={},):
        
        
        
        if self.isShow:
            
            # model_matrix_loc = locMap.get('u_ModelMatrix', -1)
            # if model_matrix_loc != -1:
            #     glUniformMatrix4fv(model_matrix_loc, 1, GL_FALSE, self.transform.T, None)

            if self.__renderType == GL_POINTS:
                loc = locMap.get('u_pointSize', -1)
                if loc != -1:
                    glUniform1f(loc, self.size)
            elif self.__renderType == GL_LINES:
                loc = locMap.get('u_lineWidth', -1)
                if loc != -1:
                    glUniform1f(loc, self.size)

            if hasattr(self, 'materialParameter'):
                for i, (name, param) in enumerate(self.materialParameter.items()):
                    loc = locMap.get(name, -1)
                    if loc != -1:
                        if param['type'] == 'float':
                            glUniform1f(loc, param['data'])
                        elif param['type'] == 'int':
                            glUniform1i(loc, param['data'])
                        
                        elif param['type'] == 'texture':
                            glActiveTexture(GL_TEXTURE0 + i)
                            glBindTexture(GL_TEXTURE_2D, param['data'])
                            glUniform1i(loc, i)
                            
                        elif param['type'] == 'vec3':
                            glUniform3fv(loc, 1, param['data'], None)
                            
                        elif param['type'] == 'vec4':
                            glUniform4fv(loc, 1, param['data'], None)
                            
                        elif param['type'] == 'vec2':
                            glUniform2fv(loc, 1, param['data'], None)

            self.vao.bind()
            
            ebolen = self.vao.getEBOLen()
            if not ebolen:
                glDrawArrays(self.__renderType, 0, self.vao.getVertexLen())
            else:
                glDrawElements(self.__renderType, ebolen, GL_UNSIGNED_INT, None)
            self.vao.unbind()



class UnionObject(BaseObject):
    def __init__(self) -> None:
        super().__init__()
        
        self.objs:List[BaseObject] = []
        
    @property
    def mainColors(self):
        if len(self.objs):
            return self.objs[0].mainColors
        else:
            return self.__mainColors
        
    def add(self, obj:BaseObject):
        self.objs.append(obj)
        if not isinstance(obj, Mesh):
            self.size = obj.size
            
        self.transform = obj.transform
    
    def load(self):
        for obj in self.objs:
            obj.load()
        
    def render(self, **kwargs):
        for obj in self.objs:
            obj.render(**kwargs)
            
    def setMultiProp(self, props:dict):
        for obj in self.objs:
            obj.setMultiProp(props)
                
    def setTransform(self, transform:np.ndarray):
        for obj in self.objs:
            obj.setTransform(transform)
            
    def cleanup(self):
        for obj in self.objs:
            obj.cleanup()

            

class PointCloud(BaseObject):
    def __init__(self, vertex:np.ndarray, 
                 color:Optional[np.ndarray|list|tuple]=DEFAULT_COLOR4, 
                 norm:Optional[np.ndarray]=None, 
                 size:int=3, 
                 transform:Optional[np.ndarray]=None) -> None:
        
        super().__init__()        
        self.size = size
        self.renderType = GL_POINTS
        self.transform = transform
    

        self.vertex = vertex.reshape(-1, 3)
        
        self.color = self.checkColor(self.vertex.shape, color)

        self.mainColors, per = colorManager.extract_dominant_colors(self.color, n_colors=3)

    def load(self):
        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.vertex, 3)
        self.vao.bindVertexBuffer(1, self.color, 4)

        self.setContextfromCurrent()


class Arrow(BaseObject):
    def __init__(self, vertex:np.ndarray, indices:np.ndarray, normal:Optional[np.ndarray]=None, color:Optional[np.ndarray|list|tuple]=DEFAULT_COLOR4, size:int=3, transform:Optional[np.ndarray]=None) -> None:
        super().__init__()
        self.vertex = vertex
        self.color = self.checkColor(self.vertex.shape, color)
        self.mainColors = colorManager.extract_dominant_colors(self.color, n_colors=3)[0]
        self.renderType = GL_TRIANGLES
        self.size = size
        self.transform = transform
        self.normal = normal
        self.indices = indices
        self.materialParameter = BaseObject.getDefaultMaterialParameter()
        

    @staticmethod
    def getTemplate(size=0.001) -> np.ndarray:

        # vertex = np.array([
        #     [0, 0, 1],
        #     [0.5, 0.5, 0],
        #     [0.5, -0.5, 0],
            
        #     [0, 0, 1],
        #     [-0.5, 0.5, 0],
        #     [0.5, 0.5, 0],
            
        #     [0, 0, 1],
        #     [-0.5, 0.5, 0],
        #     [-0.5, -0.5, 0],

        #     [0, 0, 1],
        #     [-0.5, -0.5, 0],
        #     [0.5, -0.5, 0],
        # ])*size
        vertices = np.array([
            [ 0.0000,  0.0000,  0.0000],  # Top (0)
            [ 0.5000,  0.0000,  -1.0000],  # Bottom Point 0 (1)
            [ 0.4330,  0.2500,  -1.0000],  # (2)
            [ 0.2500,  0.4330,  -1.0000],  # (3)
            [ 0.0000,  0.5000,  -1.0000],  # (4)
            [-0.2500,  0.4330,  -1.0000],  # (5)
            [-0.4330,  0.2500,  -1.0000],  # (6)
            [-0.5000,  0.0000,  -1.0000],  # (7)
            [-0.4330, -0.2500,  -1.0000],  # (8)
            [-0.2500, -0.4330,  -1.0000],  # (9)
            [ 0.0000, -0.5000,  -1.0000],  # (10)
            [ 0.2500, -0.4330,  -1.0000],  # (11)
            [ 0.4330, -0.2500,  -1.0000],  # (12)
            [ 0.0000,  0.0000,  -1.0000],  # Bottom center (13)
        ], dtype=np.float32)*size
        
        normals = np.array([
            [ 0.0000,  0.0000,  1.0000],  # 0:
            [ 1.0000,  0.0000,  0.0000],
            [ 0.8660,  0.5000,  0.0000],
            [ 0.5000,  0.8660,  0.0000],
            [ 0.0000,  1.0000,  0.0000],
            [-0.5000,  0.8660,  0.0000],
            [-0.8660,  0.5000,  0.0000],
            [-1.0000,  0.0000,  0.0000],
            [-0.8660, -0.5000,  0.0000],
            [-0.5000, -0.8660,  0.0000],
            [ 0.0000, -1.0000,  0.0000],
            [ 0.5000, -0.8660,  0.0000],
            [ 0.8660, -0.5000,  0.0000],
            [ 0.0000,  0.0000, -1.0000],
        ], dtype=np.float32)
        
        indices = np.array([
            [0,  1,  2 ],
            [0,  2,  3 ],
            [0,  3,  4 ],
            [0,  4,  5 ],
            [0,  5,  6 ],
            [0,  6,  7 ],
            [0,  7,  8 ],
            [0,  8,  9 ],
            [0,  9,  10],
            [0, 10, 11],
            [0, 11, 12],
            [0, 12, 1 ],
            
            [13,  2,  1 ],
            [13,  3,  2 ],
            [13,  4,  3 ],
            [13,  5,  4 ],
            [13,  6,  5 ],
            [13,  7,  6 ],
            [13,  8,  7 ],
            [13,  9,  8 ],
            [13, 10,  9 ],
            [13, 11, 10 ],
            [13, 12, 11 ],
            [13,  1, 12 ],
        ], dtype=np.uint32)

        return vertices, normals, indices
        
    @staticmethod
    def transformTemplate(vertex:np.ndarray, R:np.ndarray, T:np.ndarray) -> np.ndarray:
        '''
        R: (B, 3, 3)
        T: (B, 3)
        '''
        vertex = R @ vertex.T
        vertex = vertex.T + T
        
        return vertex
        
        
    def load(self, ):

        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.vertex, 3)
        self.vao.bindVertexBuffer(1, self.color, 4)
        if self.normal is not None:
            self.vao.bindVertexBuffer(2, self.normal, 3)
        self.vao.bindElementBuffer(self.indices)
        
        self.setContextfromCurrent()

class Grid(BaseObject):
    def __init__(self, n:int=51, scale:float=1.0) -> None:
        super().__init__()
        z = 0
        N = n
        NUM = (2*N+1)

        x = np.linspace(-N, N, NUM)
        y = np.linspace(-N, N, NUM)
        
        xv, yv = np.meshgrid(x, y)

        lineX = np.array(
            [[[0, 0, z, 0, 0, z]]]
        ).repeat(NUM, 1).repeat(NUM, axis=0)

        # lineX = lineX[None, :, :]

        lineX[:, :, 0] = yv 
        lineX[:, :, 3] = yv                                             
        lineX[:, :, 1] = xv
        lineX[:, :, 4] = xv + 1
        

        lineY = np.array(
            [[[0, 0, z, 0, 0, z]]]
        ).repeat(NUM, 1).repeat(NUM, axis=0)

        # lineX = lineX[None, :, :]

        lineY[:, :, 0] = xv 
        lineY[:, :, 3] = xv + 1
        lineY[:, :, 1] = yv
        lineY[:, :, 4] = yv     

        colorX = np.ones((*lineX.shape[:2], 8), dtype=np.float32) * np.array([[[0.35, 0.35, 0.35, .8]*2]])
        colorX1 = np.ones((*lineX.shape[:2], 8), dtype=np.float32) * np.array([[[0.35, 0.35, 0.35, .8]*2]])
        colorY = np.ones((*lineY.shape[:2], 8), dtype=np.float32) * np.array([[[0.35, 0.35, 0.35, .8]*2]])

        colorY[51, :, :] = np.array([[0.69, 0.18, 0.32, 0.9]*2])
        colorX[51, :, :] = np.array([[0.53, 0.76, 0.45, 0.9]*2])
        colorX1[51, :, :] = np.array([[0.01, 0.30, 0.67, 0.9]*2])

        lineX = lineX.reshape(-1, 3)
        lineY = lineY.reshape(-1, 3)
        
        
        
        self.vertex = np.concatenate((lineX, lineY), 0)
        self.norm = np.zeros_like(self.vertex)
        self.norm[:, 2] = 1.0        
        # self.color = np.array([[0.35, 0.35, 0.35, .8]]).repeat(self.vertex.shape[0], 0)
        self.color = np.concatenate((colorX.reshape(-1, 4), colorY.reshape(-1, 4)), 0)
        self.color1 = np.concatenate((colorX1.reshape(-1, 4), colorY.reshape(-1, 4)), 0)
        self.color2 = np.concatenate((colorX.reshape(-1, 4), colorX1.reshape(-1, 4)), 0)

        self.transformList = [
            np.array([[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]], dtype=np.float32),
            np.array([[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]], dtype=np.float32),

            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
        ]
        
        self.colorList = [
            self.color2,
            self.color2,
            self.color1,
            self.color1,
            self.color,
            self.color,
            self.color,
        ]
        
        self.mode = 6
        
        self.vertex = self.vertex * scale

    def setMode(self, mode:int):
        assert mode in range(7), 'mode should be in [0, 6]'
        if mode != self.mode:
            self.mode = mode
            self.transform = self.transformList[mode]
            if self.vao.getVAO() != 0:
                if self.makeCurrent():
                    self.vao.bindVertexBuffer(1, self.colorList[mode], 4)

    def load(self, ):
        
        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.vertex, 3)
        self.vao.bindVertexBuffer(1, self.colorList[self.mode], 4)
        self.vao.bindVertexBuffer(2, self.norm, 3)
        
        self.setContextfromCurrent()        

        self.size = 2
        self.renderType = GL_LINES
        self.transform = self.transformList[self.mode]


class Axis(BaseObject):
    def __init__(self, length:float=1, transform:Optional[np.ndarray]=None) -> None:
        super().__init__()
        
        self.line = np.array(
            [
                [0.000,0,0],
                [length,0,0],
                [0,0.000,0],
                [0,length, 0],
                [0,0,0.000],
                [0,0,length],
            ]
        )
            
        self.color = np.array(
            [
                [176, 48, 82, 200],
                [176, 48, 82, 200],
                [136, 194, 115, 200],
                [136, 194, 115, 200],
                [2, 76, 170, 200],
                [2, 76, 170, 200],
            ]
        ) / 255.
        
        self.transform = transform
        
        self.mainColors = colorManager.extract_dominant_colors(self.color, n_colors=3)[0]
                        
    def load(self, ):
        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.line, 3)
        self.vao.bindVertexBuffer(1, self.color, 4)
        
        self.setContextfromCurrent()
        
        self.size = 6
        self.renderType = GL_LINES


class BoundingBox(BaseObject):
    def __init__(self, vertex:np.ndarray, color:Optional[np.ndarray|list|tuple]=DEFAULT_COLOR4, norm:Optional[np.ndarray]=None, size:int=3, transform:Optional[np.ndarray]=None) -> None:
        super().__init__()

        lineIndex = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]

        self.vertex = vertex[..., lineIndex, :].reshape(-1, 3)
        
        if hasattr(color, 'shape') and color.shape[:-1] == vertex.shape[:-1]:
            color = color[..., lineIndex, :].reshape(-1, color.shape[-1])
            
        self.color = self.checkColor(self.vertex.shape, color)
        self.mainColors = colorManager.extract_dominant_colors(self.color, n_colors=3)[0]
        self.transform = transform
        self.renderType = GL_LINES
        self.size = size
        
    def load(self, ):

        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.vertex, 3)
        self.vao.bindVertexBuffer(1, self.color, 4)

        self.setContextfromCurrent()


class Lines(BaseObject):
    def __init__(self, vertex:np.ndarray, color:Optional[np.ndarray|list|tuple]=DEFAULT_COLOR4, norm:Optional[np.ndarray]=None, size:int=3, transform:Optional[np.ndarray]=None) -> None:
        super().__init__()

        self.transform = transform

        
        self.vertex = vertex.reshape(-1, 3)
        self.color = self.checkColor(self.vertex.shape, color)
                        
        self.renderType = GL_LINES
        self.size = size
        
        self.mainColors = colorManager.extract_dominant_colors(self.color, n_colors=3)[0]
        
        
    def load(self, ):
        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.vertex, 3)
        self.vao.bindVertexBuffer(1, self.color, 4)
        
        self.setContextfromCurrent()

class Mesh(BaseObject):

    def __init__(self, vertex:np.ndarray, 
                 indices:np.ndarray, 
                 color:Optional[np.ndarray|list|tuple]=DEFAULT_COLOR4, 
                 norm:Optional[np.ndarray]=None, 
                 texture:Optional[SimpleMaterial|PBRMaterial]=None, 
                 texcoord:Optional[np.ndarray]=None, 
                 faceNorm:Optional[np.ndarray]=False, 
                 transform:Optional[np.ndarray]=None) -> None:
        super().__init__()
        
        self.vertex = vertex.reshape(-1, 3)
        self.indices = indices.ravel()
        
        if norm is None:
            norm = BaseObject.buildfaceNormal(self.vertex, self.indices)
            
        else:
            if faceNorm:
                norm = BaseObject.faceNormal2VertexNormal(norm.repeat(3, axis=0), self.vertex, self.indices)



        # self.uv = texcoord.reshape(-1, 2) if isinstance(texcoord, np.ndarray) else None
        self.color = self.checkColor(vertex.shape, color)
        self.norm = norm.reshape(-1, 3)
        self.material = texture
        self.renderType = GL_TRIANGLES
        self.transform = transform

        self.materialParameter = BaseObject.getDefaultMaterialParameter()

        self.mainColors = colorManager.extract_dominant_colors(self.color, n_colors=3)[0]


        if isinstance(texcoord, np.ndarray):
            texcoord[:,1] = 1. - texcoord[:,1]
            
        self.texcoord = texcoord

        
    def load(self):
        
        if self.material is not None:
            self.materialParameter, miscParamter = BaseObject._parseMaterial(self.material, texcoord=self.texcoord)

            if 'u_AlbedoTexture' in self.materialParameter.keys():
                
                if miscParamter.get('mainColors') is not None:
                    self.mainColors = miscParamter['mainColors']['data']

                
            else:
                vertexColor = miscParamter['vertexColor']['data']
                # use vertexColor from material instead of color
                if isinstance(vertexColor, np.ndarray):
                    self.color = self.checkColor(self.vertex.shape, vertexColor)
                    self.mainColors, per = colorManager.extract_dominant_colors(vertexColor, n_colors=3)
        
        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.vertex, 3)
        self.vao.bindVertexBuffer(1, self.color, 4)
        self.vao.bindVertexBuffer(2, self.norm, 3)
        if self.texcoord is not None:
            self.vao.bindVertexBuffer(3, self.texcoord, 2)

        self.vao.bindElementBuffer(self.indices.ravel())
        
        self.setContextfromCurrent()


class Sphere(BaseObject):
    def __init__(self, size:float=1.0, transform:Optional[np.ndarray]=None) -> None:
        super().__init__()        
        
        rows, cols, r = 90, 180, size
        gv, gu = np.mgrid[0.5*np.pi:-0.5*np.pi:complex(0,rows), 0:2*np.pi:complex(0,cols)]
        xs = r * np.cos(gv)*np.cos(gu)
        ys = r * np.cos(gv)*np.sin(gu)
        zs = r * np.sin(gv)
        vertex = np.dstack((xs, ys, zs)).reshape(-1, 3)
        self.vertex = np.float32(vertex)

        idx = np.arange(rows*cols).reshape(rows, cols)
        idx_a, idx_b, idx_c, idx_d = idx[:-1,:-1], idx[1:,:-1], idx[:-1, 1:], idx[1:,1:]
        self.indices = np.uint32(np.dstack((idx_a, idx_b, idx_c, idx_c, idx_b, idx_d)).ravel())

        primitive = self.vertex[self.indices]
        a = primitive[::3]
        b = primitive[1::3]
        c = primitive[2::3]
        normal = np.repeat(np.cross(b-a, c-a), 3, axis=0)

        idx_arg = np.argsort(self.indices)
        rise = np.where(np.diff(self.indices[idx_arg])==1)[0]+1
        rise = np.hstack((0, rise, len(self.indices)))
 
        tmp = np.zeros((rows*cols, 3), dtype=np.float32)
        for i in range(rows*cols):
            tmp[i] = np.sum(normal[idx_arg[rise[i]:rise[i+1]]], axis=0)

        normal = tmp.reshape(rows,cols,-1)
        normal[:,0] += normal[:,-1]
        normal[:,-1] = normal[:,0]
        normal[0] = normal[0,0]
        normal[-1] = normal[-1,0]
        
        self.normal = normal.reshape(-1, 3)
        print(normal.shape, vertex.shape, self.indices.shape)
    
        # 生成纹理坐标
        u, v = np.linspace(0, 1, cols), np.linspace(0, 1, rows)
        self.texcoord = np.float32(np.dstack(np.meshgrid(u, v)).reshape(-1, 2))

        self.color = self.checkColor(vertex.shape, DEFAULT_COLOR4)

        self.transform = transform
        self.renderType = GL_TRIANGLES
        
        self.mainColors = colorManager.extract_dominant_colors(self.color, n_colors=3)[0]


    def load(self):
        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, self.vertex, 3)
        self.vao.bindVertexBuffer(1, self.color, 4)
        self.vao.bindVertexBuffer(2, self.normal, 3)
        self.vao.bindVertexBuffer(3, self.texcoord, 2)
        self.vao.bindElementBuffer(self.indices)
        
        self.setContextfromCurrent()

class FullScreenQuad(BaseObject):
    def __init__(self):
        super().__init__()


        vertices = np.array([
            [-1.0, -1.0,   0.0],
            [ 1.0, -1.0,   0.0],
            [ 1.0,  1.0,   0.0],
            [-1.0,  1.0,   0.0],
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,
            0, 2, 3,
        ], dtype=np.uint32)

        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, vertices, 3)
        self.vao.bindElementBuffer(indices)
        
        self.renderType = GL_TRIANGLES
        
        self.setContextfromCurrent()


class Character(BaseObject):
    
    def __init__(self, char:str, 
                 face:freetype.Face, 
                 pAdvance:float=0.0, 
                 color:np.ndarray=np.array([0.8, 0.8, 0.8], dtype=np.float32),
                 position:np.ndarray=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 fontSize:float=1.0) -> None:
        super().__init__()

        face.load_char(char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)

        sizeX, sizeY = face.glyph.bitmap.width, face.glyph.bitmap.rows
        bearingX, bearingY = face.glyph.bitmap_left, face.glyph.bitmap_top
        self.advance = face.glyph.advance.x >> 6

        self.materialParameter = BaseObject.getDefaultMaterialParameter()
        
        self.materialParameter.update(
            {
                'u_bearingAndSize': {'data': np.array([bearingX, bearingY, sizeX, sizeY], dtype=np.float32), 'type': 'vec4'},
                'u_advance': {'data': pAdvance, 'type': 'float'},
                'u_fontSize': {'data': fontSize, 'type': 'float'},
                'u_textColor': {'data': color, 'type': 'vec3'}
            }
        )

        
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face.glyph.bitmap.width, face.glyph.bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE, face.glyph.bitmap.buffer)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.materialParameter.update({'u_AlbedoTexture': {'data':tid, 'type':'texture'},
                                'u_EnableAlbedoTexture': {'data': 1, 'type': 'int'}})

        self.renderType = GL_POINTS
        
        self.vao.createVAO()
        self.vao.bindVertexBuffer(0, position, 3)

        self.setContextfromCurrent()

    def getAdvance(self) -> float:
        return self.advance

class Label(BaseObject):
    
    face = freetype.Face('ui/SourceCodePro-Semibold.ttf')

    def __init__(self, 
                 string:str, 
                 position:np.ndarray=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 color:np.ndarray=np.array([0.8, 0.8, 0.8], dtype=np.float32),
                 fontSize:float=1.0) -> None:
        super().__init__()

        self.renderType = GL_POINTS
        self.string = string
        self.characters:List[Character] = []
        
        self.color = color
        self.position = position
        self.fontSize = fontSize
        
        
    def load(self):
        
        self.face.set_pixel_sizes(0, int(32*self.fontSize))
        
        pAdvance = 0.0
        for char in self.string:
            chObj = Character(char, self.face, pAdvance, self.color, self.position, 1.0)
            pAdvance += chObj.getAdvance()
            self.characters.append(chObj)
            
        self.setContextfromCurrent()

    def render(self, locMap:dict={},):
        for char in self.characters:
            char.render(locMap)
            


    
    
