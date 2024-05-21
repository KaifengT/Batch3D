import os
import time
from OpenGL.GL import *
from OpenGL.arrays import vbo
import PIL.Image as im
import numpy as np
import pickle

class OBJ:
    generate_on_init = True
    @classmethod
    def loadTexture(cls, imagefile):

        print("load texture file:")
        print(imagefile)

        surf = im.open(imagefile)
        image = surf.convert('RGBA').tostring('raw', 'RGBA')

        # surf = pygame.image.load(imagefile)
        # image = pygame.image.tostring(surf, 'RGBA', 1)
        ix, iy = surf.get_rect().size

        print('ix', ix, 'iy', iy)

        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        return texid

    @classmethod
    def loadMaterial(cls, filename):
        contents = {}
        mtl = None
        dirname = os.path.dirname(filename)


        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                # load the texture referred to by this declaration
                mtl[values[0]] = values[1]
                imagefile = os.path.join(dirname, mtl['map_Kd'])
                mtl['texture_Kd'] = cls.loadTexture(imagefile)
            else:
                mtl[values[0]] = list(map(float, values[1:]))

        # print(contents)
        return contents

    def __init__(self, filename, swapyz=True, color=None, transform=np.identity(4)):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.gl_list = 0
        self.mtl = None
        self.l = 0
        self.transform = transform
        dirname = os.path.dirname(filename)

        material = None

        name = os.path.splitext(filename)[0]
        if os.path.exists(name + '.cug'):
            cached = True
        else:
            cached = False

        if cached:
            # print('\033[92m[SUCCESS]\033[0m Reading Cached file', name + '.cug')
            with open(name + '.cug', 'rb') as f:
                arrays = pickle.load(f)
            self.normalArray = arrays['normal']
            self.vertexArray = arrays['vertex']
            self.colorArray = arrays['color']
            self.previewColorArray = arrays['preview_color']
            self.vboarray = arrays['vboarray']
            self.l = len(self.vertexArray)//3

        else:
            for line in open(filename, "r"):
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                
                if values[0] == 'v':
                    v = list(map(float, values[1:4]))
                    if swapyz:
                        v = v[0], -v[2], v[1]
                    self.vertices.append(v)
                elif values[0] == 'vn':
                    v = list(map(float, values[1:4]))
                    if swapyz:
                        v = v[0], -v[2], v[1]
                    self.normals.append(v)
                elif values[0] == 'vt':
                    self.texcoords.append(list(map(float, values[1:3])))
                elif values[0] in ('usemtl', 'usemat'):
                    material = values[1]
                elif values[0] == 'mtllib':
                    print('Load OBJ texture file:', os.path.join(dirname, values[1]))
                    self.mtl = self.loadMaterial(os.path.join(dirname, values[1]))

                elif values[0] == 'f':
                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:]:
                        w = v.split('/')
                        face.append(int(w[0]))
                        if len(w) >= 2 and len(w[1]) > 0:
                            texcoords.append(int(w[1]))
                        else:
                            texcoords.append(0)
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]))
                        else:
                            norms.append(0)
                    self.faces.append((face, norms, texcoords, material))
            
            if self.generate_on_init:
                self.genVertexArray()
                print('\033[93m[SUCCESS]\033[0m Creating Cache file', name + '.obj')
                with open(name + '.cug', 'wb') as f:
                    pickle.dump({
                        'normal':self.normalArray,
                        'vertex':self.vertexArray,
                        'color':self.colorArray,
                        'preview_color':self.previewColorArray,
                        'vboarray':self.vboarray,
                    }, f)
            # self.generate(color)
        
        
        self.initVBO()
            

    def generate(self, color=None):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            if self.mtl is None:
                # glColor4f(0.8,0.8,0.8,0.9)
                glColor3ub(210,210,210)
            else:
                mtl = self.mtl[material]

                if 'texture_Kd' in mtl:
                    # use diffuse texmap
                    glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
                else:
                    # just use diffuse colour
                    # glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mtl['Ka'])
                    # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mtl['Kd'])
                    # glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mtl['Ks'])
                    r,g,b = mtl['Kd']
                    a = mtl['d'][0]
                    
                    # # glColor3f(r,g,b)s
                    if color is None:
                        glColor4f(r,g,b,a)
                    else:
                        glColor4f(*color)
                # glColor3f(1., 1., 1.)
                # print(*mtl['Kd'])
            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()

    def genVertexArray(self):
        self.vertexArray = []
        self.colorArray = []
        self.normalArray = []
        self.previewColorArray = []
        
        self.vboarray = []

        for face in self.faces:
            vertices, normals, texture_coords, material = face
            mtl = self.mtl[material]
            r,g,b = mtl['Kd']
            a = mtl['d'][0]

            for i in range(len(vertices)):
                self.normalArray.extend(list(self.normals[normals[i] - 1]))
                self.vertexArray.extend(list(self.vertices[vertices[i] - 1]))
                self.colorArray.extend([r,g,b,a])
                self.previewColorArray.extend([0.612, 0.675, 0.8, 1.])
                
                self.vboarray.append([r,g,b,a, *self.normals[normals[i] - 1], *self.vertices[vertices[i] - 1]])

        self.normalArray = np.array(self.normalArray, dtype=np.float32)
        self.vertexArray = np.array(self.vertexArray, dtype=np.float32)
        self.colorArray = np.array(self.colorArray, dtype=np.float32)
        self.previewColorArray = np.array(self.previewColorArray, dtype=np.float32)
        
        self.vboarray = np.array(self.vboarray, dtype=np.float32)
        self.l = len(self.vertexArray)//3
    # def render(self):
    #     glCallList(self.gl_list)
    def initVBO(self, ):
        # print('vbo')
        self.vboid = vbo.VBO(self.vboarray.flatten())
        self.l = len(self.vboarray)
        nbytes = 4
        stride = self.vboarray.shape[1] * nbytes
        self._vboInfo = {
            'vbotype':GL_C4F_N3F_V3F,
            'len':self.l,
            'nbytes':nbytes,
            'stride':stride,
        }
        
        
        self._vboMap = {
            'a_Color'   :{'size':4, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':self.vboid + 0},
            'a_Normal'  :{'size':3, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':self.vboid + 4 * nbytes},
            'a_Position':{'size':3, 'type':GL_FLOAT, 'normalized':GL_FALSE, 'stride':stride, 'pointer':self.vboid + 7 * nbytes},
        }
        self.renderType = GL_TRIANGLES

    def render(self, preview=False):
        
        # if preview:
        #     glColorPointer(4, GL_FLOAT, 0, self.previewColorArray)        
        # else:
        #     glColorPointer(4, GL_FLOAT, 0, self.colorArray)
        # glNormalPointer(GL_FLOAT, 0, self.normalArray)
        # glVertexPointer(3, GL_FLOAT, 0, self.vertexArray)
        
        # glDrawArrays(GL_TRIANGLES, 0, self.l)
        glPushMatrix()
        self.vboid.bind()
        glInterleavedArrays(GL_C4F_N3F_V3F,0,None)
        glMultMatrixf(self.transform.T)
        glDrawArrays(GL_TRIANGLES, 0, self.l)
        self.vboid.unbind()
        glPopMatrix()

    def renderinShader(self, ratio=1., locMap:dict={}):
        
        if hasattr(self, 'vboid'):
            self.vboid.bind()
            
            for attr, args in self._vboMap.items():
                loc = locMap.get(attr, None)
                if loc is not None and loc != -1:
                    glEnableVertexAttribArray(loc)
                    glVertexAttribPointer(loc, **args)
                    
            self.vboid.unbind()
                    
            glDrawArrays(self.renderType, 0, self._vboInfo['len'])

