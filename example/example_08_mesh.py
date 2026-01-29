
from b3d import b3d
import trimesh
import numpy as np

cube = trimesh.creation.box(extents=(1, 1, 1))

cube_smaller = cube.copy()
cube_smaller.apply_scale(0.5)

cube_bigger = cube.copy()
cube_bigger.apply_scale(2.0)

cube_bigger.visual.material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=np.array([253, 121, 121, 250])/255.0,
                        metallicFactor=0.0,
                        roughnessFactor=0.9,
                    )

cube_smaller.visual.material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=np.array([0, 255, 0, 100])/255.0,
                        metallicFactor=0.2,
                        roughnessFactor=0.9,
                    )


cube.visual.material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=np.array([255, 0, 0, 100])/255.0,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )


b3d.add({
    'cube': cube,
    'cube_smaller': cube_smaller,
    'cube_bigger': cube_bigger,
})