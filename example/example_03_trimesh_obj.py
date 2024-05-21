import trimesh

mesh = trimesh.load_mesh('example\Common fangtooth.glb')

print('viewer', {'ID':1, 'objType':'trimesh', 'obj':mesh})