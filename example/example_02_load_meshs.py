import trimesh
import time
import numpy as np
from b3d import b3d

mesh = trimesh.load_mesh('Common fangtooth.glb')



# Method 1: load a trimesh object
b3d.add({'trimesh object':mesh})

# Method 2: load a custom mesh with vertex and face
mesh_cus = {
    'vertex':mesh.vertices,
    'face':mesh.faces,
}
b3d.add({'custom mesh':mesh_cus})

# Method 3: directly load a obj/glb file
b3d.add('Common fangtooth.glb')

# Method 4: reset the current scene with a new obj/glb file
# b3d.updateObj('Common fangtooth.glb')

# clear all objects
# obj = b3d.getWorkspaceObj()
# b3d.rm(list(obj.keys()))

# or just call:
# b3d.clear()


b3d.GL.camera.setCamera(distance=0.2)