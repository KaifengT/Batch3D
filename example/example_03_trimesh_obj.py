import trimesh
import time
import numpy as np
mesh = trimesh.load_mesh('example\Common fangtooth.glb')

print('viewer', {'objType':'clean',})

print('viewer', {'ID':1, 'objType':'trimesh', 'obj':mesh})

# print('viewer', {'clean':True,})

def loop():

    i = np.random.randn()
    time.sleep(0.01)
    
    transform = trimesh.transformations.rotation_matrix(i*0.1, [0, 1, 0], [0, 0, 0])
    print('viewer', {'ID':1, 'transform':transform})