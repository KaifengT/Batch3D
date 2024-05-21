import numpy as np
from plyfile import PlyData
import time
from PySide6.QtWidgets import QWidget

# Specify the path to the PLY file
filepath = './example/colored-pointcloud.ply'
def npmemmap_to_array(memmap:np.memmap):
    return np.array(memmap.tolist())

# Read the PLY file
plydata = PlyData.read(filepath)

# Access the vertex data
vertices = plydata['vertex'].data

prop_list = [prop.name for prop in plydata['vertex'].properties]
if 'x' in prop_list:
    v_ind = prop_list.index('x')
    
if 'red' in prop_list:
    c_ind = prop_list.index('red')
    
if 'nx' in prop_list:
    n_ind = prop_list.index('nx')


# Convert vertex data to a NumPy array


npv = npmemmap_to_array(vertices)
v = npv[:,v_ind:v_ind+3]
c = npv[:,c_ind:c_ind+3] / 255.


# Print the resulting NumPy array
print('viewer', {'ID':1, 'objType':'p', 'vertex':v, 'color':c})

