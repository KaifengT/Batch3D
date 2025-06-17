# -*- coding: utf-8 -*-
from copy import deepcopy
import math
from typing import Any
import PIL.Image as im
import numpy as np
import time, os
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from PySide6.QtCore import (QObject, Signal, QTimer, Slot)
from ctypes import *
from .utils.transformations import invHRT, rotation_matrix, rotationMatrixY, rpy2hRT
from PIL import Image



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
            
        if hasattr(self, '_texid') and isinstance(self._texid, tuple):
            # glDeleteTextures(self._texid[0])
            ...
            # self._texid[1].delete()
            # del self._texid
        
    # def show(self, show:bool=True):
    #     self.isShow = show
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
            
    def renderinShader(self, ratio=1., locMap:dict={}, render_mode=0, size=None):
        
        if hasattr(self, '_vboid') and self.props['isShow']:
            self._vboid.bind()
            
            
            if self.props['size'] is not None:
                size = self.props['size']
            # elif size is None:
            #     size = 3
            # if size:
                if self.pointSize:
                    glPointSize(size)
                elif self.lineWidth:
                    glLineWidth(size)

            else:
                if ratio > 3: ratio = 3
                if self.pointSize:
                    glPointSize(self.pointSize * ratio)
                elif self.lineWidth:
                    glLineWidth(self.lineWidth * ratio)

            
            for attr, args in self._vboMap.items():
                loc = locMap.get(attr, None)
                if loc is not None and loc != -1:
                    glEnableVertexAttribArray(loc)
                    glVertexAttribPointer(loc, **args)
            
            self._vboid.unbind()
                   
            
            if hasattr(self, '_texid') and isinstance(self._texid, tuple) and render_mode == 3:
                textureSampler, texcoordid = self._texid
                
                # print('tttttt')
                # loc = locMap.get('a_Texcoord', None)
                # texcoordid.bind()
                # glEnableVertexAttribArray(loc)
                # glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 2*4, texcoordid)
                # texcoordid.unbind()

                loc = locMap.get('u_Texture', None)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, textureSampler)
                glUniform1i(loc, 0)
                
                loc = locMap.get('render_mode', None)
                glUniform1i(loc, int(render_mode))
                
            elif render_mode == 3:
                loc = locMap.get('render_mode', None)
                glUniform1i(loc, 1)
                
            else:
                loc = locMap.get('render_mode', None)
                glUniform1i(loc, int(render_mode))
            
            
                  
            if hasattr(self, '_indid') and self._indid is not None:
                self._indid.bind()
                # if render_mode == 0:
                #     glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                #     glDrawElements(GL_LINES, self._vboInfo['len_ind'], GL_UNSIGNED_INT, None)
                # else:
                #     glDrawElements(self.renderType, self._vboInfo['len_ind'], GL_UNSIGNED_INT, None)
                if render_mode == 0:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                else:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
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
    def buildVBO(vertex, color=None, norm=None, indices=None, texture=None, texcoord=None):    
        
        # print('buildVBO vertex:', vertex.shape)
        
            
        assert hasattr(vertex, 'shape') and len(vertex.shape)==2 and vertex.shape[1]==3, 'vertex format error'
        
        if norm is not None:
            assert hasattr(norm, 'shape') and norm.ravel().shape==vertex.ravel().shape, 'norm format error'
            normArray = norm
        else:
            normArray = np.zeros_like(vertex)
            # normArray = np.zeros_like(vertex) + np.array([[0.,0.,1.]])
    
        
        
        
        
        if color is None:
            color = [0.6, 0.6, 0.6, 1.]
        
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
                colorArray = np.array([0.8,0.8,0.8, 1.]).repeat(vertex.shape[0], 0)
        
        vboArray = np.concatenate((colorArray, normArray, vertex), axis=1, dtype=np.float32)
                
        
                
        nbytes = 4
        
        vboInfo = {}
        
                
        indid = None
        texid = None
        
        if indices is not None:
            indices = np.int32(indices.ravel())
            indid = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)
            len_ind = len(indices)
            vboInfo.update({'len_ind':len_ind})

        if texcoord is not None and texture is not None:
            # print('texcoord:', texcoord.shape, 'texture:', texture)
            # texture = texture.transpose(Image.FLIP_LEFT_RIGHT)
            # texture = texture.transpose(Image.FLIP_TOP_BOTTOM)
            # texture = texture.transpose(Image.ROTATE_180)
            # 
            # texture.save('temp.png')
            textureSampler = BaseObject.createTexture2d(texture)
            
            #/root/Workspace/tkf_exp/my_sdfusion/data/ShapeNet/ShapeNetCore.v1/02942699/17a010f0ade4d1fd83a3e53900c6cbba
            texcoord[:,1] = 1. - texcoord[:,1]
            
            # texcoordid = vbo.VBO(texcoord)
            # texid = (textureSampler, texcoordid)
            texid = (textureSampler, None)
            
            vboArray = np.concatenate((texcoord, vboArray), axis=1, dtype=np.float32)
            
        vboid = vbo.VBO(vboArray)
        stride = vboArray.shape[1] * nbytes
        
        if texcoord is not None and texture is not None:
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

        
        return vboid, vboArray, vboInfo, vboMap, indid, texid

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

class PointCloud(BaseObject):
    def __init__(self, vertex:np.ndarray, color=[1., 0., 0.], norm=None, size=3, transform=None) -> None:
        super().__init__()
        
                
        self.reset()
        
        self.setTransform(transform)
        # vboArray, self.vbotype, self.l = BaseObject.buildVBO(vertex, color, norm)
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(vertex, color, norm)
    
        # self._vboid = vbo.VBO(vboArray.flatten())
        
        self.pointSize = size
        self.renderType = GL_POINTS
        # self.color = color

class Arrow(BaseObject):
    def __init__(self, vertex:np.ndarray=None, color=[1., 0., 0.], size=3, transform=None) -> None:
        super().__init__()
        self.load(vertex, color, size, transform)
        self.color = color
        self.renderType = GL_TRIANGLES
        
    @staticmethod
    def getTemplate():
        
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
        ])*0.001
        
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
        # ])*0.001
        
        # vertex = transform[:3,:3] @ vertex.T
        # vertex = vertex.T + transform[:3,3]
        # pcd = pcd.T
        self.reset()
        
        self.setTransform(transform)
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(vertex, color)
        
        # if hasattr(color, 'shape') and color.shape==vertex.shape:
        #      vboArray = np.concatenate((color, vertex), axis=1, dtype=np.float32)
        # else:
        #     colorArray = np.array([color]).repeat(vertex.shape[0], 0)
        #     vboArray = np.concatenate((colorArray, vertex), axis=1, dtype=np.float32)
        
        # self.reset()
        # self.l = len(vboArray)
        # self._vboid = vbo.VBO(vboArray.flatten())
        
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
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(line, color)
        
        # print('build vbo time:', time.time()-t)

        self.lineWidth = 2
        self.renderType = GL_LINES

class Grid(BaseObject):
    def __init__(self) -> None:
        super().__init__()
        z = 0
        N = 51


        x = np.linspace(-N, N, 2*N+1)
        y = np.linspace(-N, N, 2*N+1)

        xv, yv = np.meshgrid(x, y)

        lineX = np.array(
            [[[0, 0, z, 0, 0, z]]]
        ).repeat(2*N+1, 1).repeat(2*N+1, axis=0)

        # lineX = lineX[None, :, :]

        lineX[:, :, 0] = yv 
        lineX[:, :, 3] = yv                                             
        lineX[:, :, 1] = xv
        lineX[:, :, 4] = xv + 1        
        

        lineY = np.array(
            [[[0, 0, z, 0, 0, z]]]
        ).repeat(2*N+1, 1).repeat(2*N+1, axis=0)

        # lineX = lineX[None, :, :]

        lineY[:, :, 0] = xv 
        lineY[:, :, 3] = xv + 1                                             
        lineY[:, :, 1] = yv
        lineY[:, :, 4] = yv     


        lineX = lineX.reshape(-1, 3)
        lineY = lineY.reshape(-1, 3)
        
        line = np.concatenate((lineX, lineY), 0)
        
        color = [0.2, 0.2, 0.2, .7]
        
        self.reset()
        
        # t = time.time()
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(line, color)
        
        # print('build vbo time:', time.time()-t)

        self.lineWidth = 2
        self.renderType = GL_LINES



    
class Axis(BaseObject):
    def __init__(self, R=None, T=None, length = 1, transform=None) -> None:
        super().__init__()
        
        
        line = np.array(
            [
                [0.001,0,0],
                [length,0,0],
                [0,0.001,0],
                [0,length, 0],
                [0,0,0.001],
                [0,0,length],
            ]
        )
        
        
        
        if isinstance(R, np.ndarray):
            assert R.shape == (3, 3), 'R shape error'
            
            line[1, :] = R[:, 0] * length
            line[3, :] = R[:, 1] * length
            line[5, :] = R[:, 2] * length
            
        if isinstance(T, np.ndarray):
            
            
            line[:, 0] += T[0]
            line[:, 1] += T[1]
            line[:, 2] += T[2]
            
            
        color = np.array(
            [
                [176, 48, 82, 200],
                [176, 48, 82, 200],
                [136, 194, 115, 200],
                [136, 194, 115, 200],
                [2, 76, 170, 200],
                [2, 76, 170, 200],
            ]
        ) / 255.
        # vboArray = np.concatenate((color, line), axis=1, dtype=np.float32)
        
        self.reset()
        
        self.setTransform(transform)
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(line, color, )
        
        self.renderType = GL_LINES
        
        self.lineWidth = 8

class BoundingBox(BaseObject):
    def __init__(self, vertex:np.ndarray, color=[1., 0., 0.], norm=None, size=3, transform=None) -> None:
        super().__init__()

        lineIndex = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]

        lineArray = vertex[..., lineIndex, :].reshape(-1, 3)
        self.reset()
        
        self.setTransform(transform)
        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(lineArray, color, norm)
        
        self.renderType = GL_LINES
        self.lineWidth = size

class Lines(BaseObject):
    def __init__(self, vertex:np.ndarray, color=[1., 0., 0.], norm=None, size=3, transform=None) -> None:
        super().__init__()

        self.reset()
        self.setTransform(transform)
        # assert hasattr(vertex, 'shape') and len(vertex.shape)==3 and vertex.shape[1]==2 and vertex.shape[2]==3, 'line vertex format error, must be (n, 2, 3)'
        vertex = vertex.reshape(-1, 3)
        # vboArray, self.vbotype, self.l = BaseObject.buildVBO(vertex, color, norm)
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(vertex, color, norm)
        
        # self.l = len(vboArray)
        # self._vboid = vbo.VBO(vboArray.flatten())
        self.renderType = GL_LINES
        self.lineWidth = size

class Mesh(BaseObject):

    def __init__(self, vertex:np.ndarray, indices=None, color=[0.6, 0.6, 0.6], norm=None, texture=None, texcoord=None, faceNorm=False, transform=None) -> None:
        super().__init__()

        self.reset()
        
        self.setTransform(transform)
        # print(normal.shape, vs.shape, indices.shape)
    
        # # 生成纹理坐标
        # u, v = np.linspace(0, 1, cols), np.linspace(0, 1, rows)
        # texcoord = np.float32(np.dstack(np.meshgrid(u, v)).reshape(-1, 2))
        indices = np.int32(indices.ravel())
        
        if norm is None:
            norm = BaseObject.buildfaceNormal(vertex, indices)
            
        else:
            if faceNorm:
                norm = BaseObject.faceNormal2VertexNormal(norm.repeat(3, axis=0), vertex, indices)

        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(vertex, 
                                                                                                           color, 
                                                                                                           norm, 
                                                                                                           indices,
                                                                                                           texture,
                                                                                                           texcoord)
            
        
        self.renderType = GL_TRIANGLES

class Sphere(BaseObject):

    def __init__(self, vertex:np.ndarray, indices=None, color=[1., 0., 0.], norm=None, texture=None) -> None:
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

        
        self._vboid, vboArray, self._vboInfo, self._vboMap, self._indid, self._texid = BaseObject.buildVBO(vs, np.array([1., 1., 0.6, 1.]), normal, indices)
            
        
        self.renderType = GL_TRIANGLES
