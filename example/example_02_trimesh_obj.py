import trimesh
import time
import numpy as np


mesh = trimesh.load_mesh('example\Common fangtooth.glb')


mesh_cus = {
    'vertex':mesh.vertices,
    'face':mesh.faces,
}

Batch3D.addObj({'mesh':mesh,
                'mesh_cus':mesh_cus})

