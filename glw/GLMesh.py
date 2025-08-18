# -*- coding: utf-8 -*-
from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import Any
import PIL.Image as im
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from ctypes import *
from .utils.transformations import invHRT, rotation_matrix, rotationMatrixY, rpy2hRT
from PIL import Image
from trimesh.visual.material import SimpleMaterial, PBRMaterial

DEFAULT_COLOR3 = (0.8, 0.8, 0.8)
DEFAULT_COLOR4 = (0.8, 0.8, 0.8, 1.0)
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


class BaseObject:
    def __init__(self) -> None:
        self.pointSize = 0
        self.lineWidth = 0
        
        self.isShow = True
        
        self.props = {
            'isShow':True,
            'size':3
        }

        self.renderType = GL_POINTS
        self.vbotype = GL_C3F_V3F
        
        self._vboInfo = {
            'vbotype':self.vbotype,
            'len':0,
            'nbytes':4,
            'stride':0,
        }
        self._vboMap = {}
        
        self.isDefaultColor = True
        self.meanColor = None
        
        self.transform = np.eye(4, dtype=np.float32)

        self.mainColors = np.array([[*DEFAULT_COLOR3]]).astype(np.float32)
        # self.transform[:3, 3] = np.array([1., 10., 0.], dtype=np.float32)
        
    def load(self, ):
        pass
    
    def setTransform(self, transform:np.ndarray):
        if transform is None:
            return
        assert transform.shape == (4, 4), 'transform shape error'
        self.transform = transform
    
    def reset(self, ):
        if hasattr(self, '_vboid'):
            self._vboid.delete()
            del self._vboid
            
        if hasattr(self, '_indid') and self._indid is not None:
            self._indid.delete()
            del self._indid
            
        if hasattr(self, '_texid') and isinstance(self._texid, Mapping):            
            
            datas = list(self._texid.values())
            toDelete = []
            for data in datas:
                if data['type'] == 'texture':
                    toDelete.append(data['data'])
            if len(toDelete):
                glDeleteTextures(len(toDelete), toDelete)
            del self._texid

        
    def updateProps(self, props:dict):
        self.props.update(props)
    
        
    def render(self, ratio=1.):
        if hasattr(self, '_vboid') and self.props['isShow']:
            self._vboid.bind()
            glInterleavedArrays(self.vbotype,0,None)
            if ratio > 5: ratio = 5
            if self.pointSize:
                glPointSize(self.pointSize * ratio)
            elif self.lineWidth:
                glLineWidth(self.lineWidth * ratio)
                                
            else:
                glDrawArrays(self.renderType, 0, self.l)
            self._vboid.unbind()
            
    def renderinShader(self, locMap:dict={}, render_mode=0, size=None):
        
        if hasattr(self, '_vboid') and self.props['isShow']:
            
            model_matrix_loc = locMap.get('u_ModelMatrix', None)
            if model_matrix_loc is not None and model_matrix_loc != -1:
                glUniformMatrix4fv(model_matrix_loc, 1, GL_FALSE, self.transform.T, None)
            
            
            self._vboid.bind()
            
            if self.props['size'] is not None:
                size = self.props['size']
            else:
                size = 3

            glPointSize(size)
            glLineWidth(size)

            
            for attr, args in self._vboMap.items():
                loc = locMap.get(attr, None)
                if loc is not None and loc != -1:
                    glEnableVertexAttribArray(loc)
                    glVertexAttribPointer(loc, **args)
            
            self._vboid.unbind()
                   
            if hasattr(self, '_texid') and \
                isinstance(self._texid, Mapping):
                

                for i, (locName, params) in enumerate(self._texid.items()):
                    loc = locMap.get(locName, -1)
                    if loc != -1:
                        if params['type'] == 'texture':
                            if self._vboInfo['vbotype'] == GL_T2F_C4F_N3F_V3F:
                                glActiveTexture(GL_TEXTURE0 + i)
                                glBindTexture(GL_TEXTURE_2D, params['data'])
                                glUniform1i(loc, i)
                            
                        elif params['type'] == 'float':
                            glUniform1f(loc, params['data'])
                            
                        elif params['type'] == 'int':
                            glUniform1i(loc, params['data'])

                
            loc = locMap.get('render_mode', -1)
            if loc != -1:
                glUniform1i(loc, int(render_mode))
            
            
                  
            if hasattr(self, '_indid') and self._indid is not None:
                self._indid.bind()
                if render_mode == 0:
                    # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    # glDrawElements(self.renderType, self._vboInfo['len_ind'], GL_UNSIGNED_INT, None)
                    
                    # glEnable(GL_POLYGON_OFFSET_LINE)
                    # # glPolygonOffset(-1.0, -1.0)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    glDrawElements(self.renderType, self._vboInfo['len_ind'], GL_UNSIGNED_INT, None)
                    # glDisable(GL_POLYGON_OFFSET_LINE)
                    
                else:
                    # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    glPolygonMode(GL_FRONT, GL_FILL)
                    glDrawElements(self.renderType, self._vboInfo['len_ind'], GL_UNSIGNED_INT, None)
                self._indid.unbind()
            else:
                glDrawArrays(self.renderType, 0, self._vboInfo['len'])
           
    @staticmethod
    def _decode_HexColor_to_RGB(hexcolor):
        if len(hexcolor) == 6:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4))
        elif len(hexcolor) == 8:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4, 6))
        else:
            return (1., 1., 1., 1.)
 
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
        materialParamter = {'u_Metallic': {'data': 0.0, 'type': 'float'},
                            'u_Roughness': {'data': 0.0, 'type': 'float'},
                            'u_EnableAlbedoTexture': {'data': 0, 'type': 'int'},
                            'u_EnableMetallicRoughnessTexture': {'data': 0, 'type': 'int'}}

        if texcoord is None:
            # find simple color
            if isinstance(material, SimpleMaterial):
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.diffuse), 'type': 'list'}})

            elif isinstance(material, PBRMaterial) and hasattr(material, 'baseColorFactor'):
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.baseColorFactor), 'type': 'list'}})
                
                if isinstance(material.metallicFactor, float):
                    materialParamter.update({'u_Metallic': {'data': material.metallicFactor, 'type': 'float'}})
                if isinstance(material.roughnessFactor, float):
                    materialParamter.update({'u_Roughness': {'data': material.roughnessFactor, 'type': 'float'}})

            return materialParamter, miscParamter

        if isinstance(material, SimpleMaterial):

            if isinstance(material.image, im.Image):
                tid = BaseObject.createTexture2d(material.image)

                texcolor = colorManager.get_color_from_tex(material.image, texcoord)
                main_colors, per = colorManager.extract_dominant_colors(texcolor, n_colors=3)

                materialParamter.update({'u_AlbedoTexture': {'data':tid, 'type':'texture'},
                                         'u_EnableAlbedoTexture': {'data': 1, 'type': 'int'}})
                miscParamter.update({'mainColors':{'data':main_colors, 'type':'list'}})

            else:
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.diffuse), 'type': 'list'}})

            return materialParamter, miscParamter

        elif isinstance(material, PBRMaterial):
            
            if isinstance(material.baseColorTexture, im.Image):
                
                tid = BaseObject.createTexture2d(material.baseColorTexture)
                
                texcolor = colorManager.get_color_from_tex(material.baseColorTexture, texcoord)
                main_colors, per = colorManager.extract_dominant_colors(texcolor, n_colors=3)

                materialParamter.update({'u_AlbedoTexture': {'data': tid, 'type': 'texture'},\
                                        'u_EnableAlbedoTexture': {'data': 1, 'type': 'int'}})
                miscParamter.update({'mainColors': {'data': main_colors, 'type': 'list'}})

                
            else:
                miscParamter.update({'vertexColor': {'data': _safeGetArray(material.baseColorFactor), 'type': 'list'}})
                
                
            if isinstance(material.metallicRoughnessTexture, im.Image):
                tid = BaseObject.createTexture2d(material.metallicRoughnessTexture)
                materialParamter.update({'u_MetallicRoughnessTexture': {'data': tid, 'type': 'texture'},
                                         'u_EnableMetallicRoughnessTexture': {'data': 1, 'type': 'int'}})
            else:
                
                if isinstance(material.metallicFactor, float):
                    materialParamter.update({'u_Metallic': {'data': material.metallicFactor, 'type': 'float'}})
                if isinstance(material.roughnessFactor, float):
                    materialParamter.update({'u_Roughness': {'data': material.roughnessFactor, 'type': 'float'}})

            return materialParamter, miscParamter
        else:
            raise ValueError('Unknown material type')

    @staticmethod
    def buildVBO(vertex:np.ndarray, 
                 color:np.ndarray=None, 
                 norm:np.ndarray=None, 
                 indices:np.ndarray=None, 
                 texture:SimpleMaterial|PBRMaterial=None, 
                 texcoord:np.ndarray=None
                 ):
        
        # print('buildVBO vertex:', vertex.shape)
        
            
        assert hasattr(vertex, 'shape') and len(vertex.shape)==2 and vertex.shape[1]==3, 'vertex format error'
        
        if norm is not None:
            assert hasattr(norm, 'shape') and norm.ravel().shape==vertex.ravel().shape, 'norm format error'
            normArray = norm
        else:
            normArray = np.zeros_like(vertex)
            # normArray = np.zeros_like(vertex) + np.array([[0.,0.,1.]])
    
        
        
        
        
        if color is None:
            color = [*DEFAULT_COLOR4]
        
        if isinstance(color, str):
            color = BaseObject._decode_HexColor_to_RGB(color)
    
        if hasattr(color, 'shape') and color.shape[0]==vertex.shape[0] and len(color.shape)==len(vertex.shape):
            if color.shape[1]==4:
                colorArray = color
            elif color.shape[1]==3:
                colorArray = np.concatenate((color, np.ones((color.shape[0], 1), dtype=np.float32)), axis=1)
            else:
                raise ValueError('color shape error')
                
        else:
            if len(color)==4:
                colorArray = np.array([color]).repeat(vertex.shape[0], 0)
            elif len(color)==3:
                colorArray = np.array([[*color, 1.]]).repeat(vertex.shape[0], 0)
            else:
                print('color:', color)
                # raise ValueError('color shape error')
                colorArray = np.array([*DEFAULT_COLOR4]).repeat(vertex.shape[0], 0)
        
        vboArray = [colorArray, normArray, vertex]
                
        mainColor, per = colorManager.extract_dominant_colors(colorArray, n_colors=3)
                
        nbytes = 4
        
        vboInfo = {}
        
                
        indid = None
        materialParameter = {'u_Metallic': {'data': 0.0, 'type': 'float'},
                        'u_Roughness': {'data': 0.0, 'type': 'float'},
                        'u_EnableAlbedoTexture': {'data': 0, 'type': 'int'},
                        'u_EnableMetallicRoughnessTexture': {'data': 0, 'type': 'int'}}

        validTexture = False

        if indices is not None:
            indices = np.int32(indices.ravel())
            indid = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)
            len_ind = len(indices)
            vboInfo.update({'len_ind':len_ind})


        
        if texcoord is not None:
            texcoord[:,1] = 1. - texcoord[:,1]
            

        if texture is not None:
            materialParameter, miscParamter = BaseObject._parseMaterial(texture, texcoord=texcoord)

            if 'u_AlbedoTexture' in materialParameter.keys():
                vboArray.insert(0, texcoord)
                validTexture = True
                
                if miscParamter.get('mainColors') is not None:
                    mainColor = miscParamter['mainColors']['data']

                
            else:
                vertexColor = miscParamter['vertexColor']['data']
                # use vertexColor from material instead of color
                if isinstance(vertexColor, np.ndarray):
                    vertexColor = vertexColor[None].repeat(vboArray[0].shape[0], axis=0)
                    vboArray[0] = vertexColor
                    mainColor, per = colorManager.extract_dominant_colors(vertexColor, n_colors=3)
                
        

        vboArray = np.concatenate(vboArray, axis=1, dtype=np.float32)
        vboid = vbo.VBO(vboArray)
        stride = vboArray.shape[1] * nbytes
        
        if validTexture:
            vbotype = GL_T2F_C4F_N3F_V3F
            vboMap = {
                'a_Texcoord':{'size':2, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':vboid + 0},   
                'a_Color'   :{'size':4, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':vboid + 2 * nbytes},
                'a_Normal'  :{'size':3, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':vboid + (2 + 4) * nbytes},
                'a_Position':{'size':3, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':vboid + (2 + 4 + 3) * nbytes},
                  
            }
            
        else:
            vbotype = GL_C4F_N3F_V3F
            vboMap = {
                'a_Color'   :{'size':4, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':vboid + 0},
                'a_Normal'  :{'size':3, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':vboid + colorArray.shape[1] * nbytes},
                'a_Position':{'size':3, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':vboid + (colorArray.shape[1] + normArray.shape[1]) * nbytes},
            }
            
     
        vboInfo.update({
            'vbotype':vbotype,
            'len':len(vboArray),
            'nbytes':nbytes,
            'stride':stride,
        })

        
        return vboid, vboArray, vboInfo, vboMap, indid, materialParameter, mainColor

    @staticmethod
    def createTexture2d(texture_file:Image.Image):

        # im = np.array(Image.open('earth.jpg'))
        
        im = np.array(texture_file)
        im_h, im_w = im.shape[:2]
        im_mode = GL_LUMINANCE if im.ndim == 2 else (GL_RGB, GL_RGBA)[im.shape[-1]-3]

        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)

        if (im.size/im_h)%4 == 0:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        else:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
        glTexImage2D(GL_TEXTURE_2D, 0, im_mode, im_w, im_h, 0, im_mode, GL_UNSIGNED_BYTE, im)
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

        
    def __del__(self):
        self.reset()


class UnionObject(BaseObject):
    def __init__(self) -> None:
        super().__init__()
        
        self.objs = []
        
    def add(self, obj:BaseObject):
        self.objs.append(obj)
        
    def reset(self):
        for obj in self.objs:
            obj.reset()
        
    def show(self, show:bool=True):
        for obj in self.objs:
            obj.show(show)
    
    def load(self):
        for obj in self.objs:
            obj.load()
        
    def renderinShader(self, **kwargs):
        for obj in self.objs:
            obj.renderinShader(**kwargs)
            
    def updateProps(self, props:dict):
        for obj in self.objs:
            if hasattr(obj, 'props'):
                obj.props.update(props)
                
    def updateTransform(self, transform:np.ndarray):
        for obj in self.objs:
            obj.setTransform(transform)

            

class PointCloud(BaseObject):
    def __init__(self, vertex:np.ndarray, color=[*DEFAULT_COLOR4], norm=None, size=3, transform=None) -> None:
        super().__init__()
        
                
        self.reset()
        
        self.setTransform(transform)
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(vertex, color, norm)
    
        # self._vboid = vbo.VBO(vboArray.flatten())
        
        self.pointSize = size
        self.renderType = GL_POINTS
        # self.color = color

class Arrow(BaseObject):
    def __init__(self, vertex:np.ndarray=None, color=[*DEFAULT_COLOR4], size=3, transform=None) -> None:
        super().__init__()
        self.load(vertex, color, size, transform)
        self.color = color
        self.renderType = GL_TRIANGLES
        
    @staticmethod
    def getTemplate(size=0.001) -> np.ndarray:

        vertex = np.array([
            [0, 0, 1],
            [0.5, 0.5, 0],
            [0.5, -0.5, 0],
            
            [0, 0, 1],
            [-0.5, 0.5, 0],
            [0.5, 0.5, 0],
            
            [0, 0, 1],
            [-0.5, 0.5, 0],
            [-0.5, -0.5, 0],

            [0, 0, 1],
            [-0.5, -0.5, 0],
            [0.5, -0.5, 0],
        ])*size
        
        return vertex
        
    @staticmethod
    def transformTemplate(vertex:np.ndarray, R:np.ndarray, T:np.ndarray) -> np.ndarray:
        '''
        R: (B, 3, 3)
        T: (B, 3)
        '''
        vertex = R @ vertex.T
        vertex = vertex.T + T
        
        return vertex
        
        
    def load(self, vertex:np.ndarray, color=[1., 0., 0.], size=3, transform=None):
        self.reset()
        self.setTransform(transform)
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(vertex, color)
        self.pointSize = size

class Grid_old(BaseObject):
    def __init__(self) -> None:
        super().__init__()
        z = 0
        lineX = np.array(
            [[0,-100, z, 0, 100, z]]
        ).repeat(201, 0)
        xaxis = np.linspace(-100, 100, 201)
        lineY = np.array(
            [[-100, 0, z, 100, 0, z]]
        ).repeat(201, 0)
        
        lineX[:,0] = xaxis
        lineX[:,3] = xaxis
        lineX = lineX.reshape(-1, 3)
        
        lineY[:,1] = xaxis
        lineY[:,4] = xaxis
        lineY = lineY.reshape(-1, 3)
        
        line = np.concatenate((lineX, lineY), 0)
        
        color = [0.2, 0.2, 0.2, .5]
        
        self.reset()
        
        # t = time.time()
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(line, color)
        
        # print('build vbo time:', time.time()-t)

        self.lineWidth = 2
        self.renderType = GL_LINES

class Grid(BaseObject):
    def __init__(self, n=51, scale=1.0) -> None:
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


        lineX = lineX.reshape(-1, 3)
        lineY = lineY.reshape(-1, 3)
        
        self.line = np.concatenate((lineX, lineY), 0)
        self.norm = np.zeros_like(self.line)
        self.norm[:, 2] = 1.0
        # color = [0.2, 0.2, 0.2, .7]
        self.color = [0.35, 0.35, 0.35, .8]
    
        
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
        ]
        
        scaleMatrix = np.eye(4, dtype=np.float32)
        scaleMatrix[:3, :3] *= scale
        for i in range(len(self.transformList)):
            self.transformList[i] = scaleMatrix @ self.transformList[i]
            
        self.transform = self.transformList[5]
        
        
        # print('build vbo time:', time.time()-t)

        self.props['size'] = 2
        self.renderType = GL_LINES


    def manualBuild(self, ):
        self.reset()
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(self.line, self.color, self.norm)
        
class Axis(BaseObject):
    def __init__(self, R=None, T=None, length = 1, transform=None) -> None:
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
        
        
        
        if isinstance(R, np.ndarray):
            assert R.shape == (3, 3), 'R shape error'
            
            self.line[1, :] = R[:, 0] * length
            self.line[3, :] = R[:, 1] * length
            self.line[5, :] = R[:, 2] * length
            
        if isinstance(T, np.ndarray):
            
            
            self.line[:, 0] += T[0]
            self.line[:, 1] += T[1]
            self.line[:, 2] += T[2]
            
            
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
        
        
        self.setTransform(transform)
                
        self.renderType = GL_LINES
        
        self.props['size'] = 6

    def manualBuild(self, ):
        self.reset()
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(self.line, self.color)


class BoundingBox(BaseObject):
    def __init__(self, vertex:np.ndarray, color=[*DEFAULT_COLOR4], norm=None, size=3, transform=None) -> None:
        super().__init__()

        lineIndex = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]

        lineArray = vertex[..., lineIndex, :].reshape(-1, 3)
        
        if hasattr(color, 'shape') and color.shape[:-1] == vertex.shape[:-1]:
            color = color[..., lineIndex, :].reshape(-1, color.shape[-1])
        
        self.reset()
        self.setTransform(transform)
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(lineArray, color, norm)
        
        self.renderType = GL_LINES
        self.lineWidth = size

class Lines(BaseObject):
    def __init__(self, vertex:np.ndarray, color=[*DEFAULT_COLOR4], norm=None, size=3, transform=None) -> None:
        super().__init__()

        self.reset()
        self.setTransform(transform)
                
        if hasattr(color, 'shape') and color.shape[:-1] == vertex.shape[:-1]:
            color = color.reshape(-1, color.shape[-1])
            
        vertex = vertex.reshape(-1, 3)
                
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(vertex, color, norm)
        
        self.renderType = GL_LINES
        self.lineWidth = size

class Mesh(BaseObject):

    def __init__(self, vertex:np.ndarray, 
                 indices:np.ndarray=None, 
                 color=[*DEFAULT_COLOR4], 
                 norm:np.ndarray=None, 
                 texture:SimpleMaterial|PBRMaterial=None, 
                 texcoord:np.ndarray=None, 
                 faceNorm:np.ndarray=False, 
                 transform:np.ndarray=None) -> None:
        super().__init__()

        self.reset()
        
        self.setTransform(transform)

        
        
        indices = np.int32(indices.ravel())
        
        if norm is None:
            norm = BaseObject.buildfaceNormal(vertex, indices)
            
        else:
            if faceNorm:
                norm = BaseObject.faceNormal2VertexNormal(norm.repeat(3, axis=0), vertex, indices)

        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(vertex, 
                                                                                                           color, 
                                                                                                           norm, 
                                                                                                           indices,
                                                                                                           texture,
                                                                                                           texcoord)
            
        
        self.renderType = GL_TRIANGLES

class Sphere(BaseObject):

    def __init__(self, vertex:np.ndarray, indices=None, color=[*DEFAULT_COLOR4], norm=None, texture=None) -> None:
        super().__init__()

        self.reset()
        
        
        rows, cols, r = 90, 180, 1
        gv, gu = np.mgrid[0.5*np.pi:-0.5*np.pi:complex(0,rows), 0:2*np.pi:complex(0,cols)]
        xs = r * np.cos(gv)*np.cos(gu)
        ys = r * np.cos(gv)*np.sin(gu)
        zs = r * np.sin(gv)
        vs = np.dstack((xs, ys, zs)).reshape(-1, 3)
        vs = np.float32(vs)

        # 生成三角面的索引
        idx = np.arange(rows*cols).reshape(rows, cols)
        idx_a, idx_b, idx_c, idx_d = idx[:-1,:-1], idx[1:,:-1], idx[:-1, 1:], idx[1:,1:]
        indices = np.int32(np.dstack((idx_a, idx_b, idx_c, idx_c, idx_b, idx_d)).ravel())

        # 生成法向量
        primitive = vs[indices]
        a = primitive[::3]
        b = primitive[1::3]
        c = primitive[2::3]
        normal = np.repeat(np.cross(b-a, c-a), 3, axis=0)

        idx_arg = np.argsort(indices)
        rise = np.where(np.diff(indices[idx_arg])==1)[0]+1
        rise = np.hstack((0, rise, len(indices)))
 
        tmp = np.zeros((rows*cols, 3), dtype=np.float32)
        for i in range(rows*cols):
            tmp[i] = np.sum(normal[idx_arg[rise[i]:rise[i+1]]], axis=0)

        normal = tmp.reshape(rows,cols,-1)
        normal[:,0] += normal[:,-1]
        normal[:,-1] = normal[:,0]
        normal[0] = normal[0,0]
        normal[-1] = normal[-1,0]
        
        normal = normal.reshape(-1, 3)
        print(normal.shape, vs.shape, indices.shape)
    
        # 生成纹理坐标
        u, v = np.linspace(0, 1, cols), np.linspace(0, 1, rows)
        texcoord = np.float32(np.dstack(np.meshgrid(u, v)).reshape(-1, 2))

        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(vs, np.array([1., 1., 0.6, 1.]), normal, indices)
            
        
        self.renderType = GL_TRIANGLES


class FullScreenQuad(BaseObject):
    def __init__(self):
        super().__init__()


        vertices = np.array([
            [-1.0, -1.0,   0.0],
            [ 1.0, -1.0,   0.0],
            [ 1.0,  1.0,   0.0],
            [-1.0,  1.0,   0.0],
        ], dtype=np.float32)

        # 索引（两个三角形）
        indices = np.array([
            0, 1, 2,
            0, 2, 3,
        ], dtype=np.uint32)

        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid, self.mainColors = BaseObject.buildVBO(vertex=vertices, indices=indices)
        
        self.renderType = GL_TRIANGLES

