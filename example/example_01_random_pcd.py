
import numpy as np
import os, sys
import pickle
import time
import struct
from b3d import b3d



pcd = np.random.rand(20000, 3) 
pcd2 = np.random.rand(20000, 6) 

lines = np.random.rand(100, 2, 3)
bboxs = np.random.rand(8, 3)

b3d.add({'pointcloud_#FF0000DD_&10':pcd,
                'colored_pcd':pcd2,
                '2_line_&10':lines,
                '3_bbox_&10':bboxs,
                'axis_&5': np.eye(4)*10,})



    
    
