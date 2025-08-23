import trimesh
import time
import numpy as np


mesh = trimesh.load_mesh('Common fangtooth.glb')


mesh_cus = {
    'vertex':mesh.vertices,
    'face':mesh.faces,
}

b3d.addObj({'mesh':mesh,
                'mesh_cus':mesh_cus})

