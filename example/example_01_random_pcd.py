
import numpy as np
import os, sys
import pickle
import time
import struct




pcd = np.random.rand(20000, 3) 
pcd2 = np.random.rand(20000, 6) 

lines = np.random.rand(100, 2, 3)
bboxs = np.random.rand(8, 3)

Batch3D.addObj({'pcd':pcd,
                'colored_pcd':pcd2,
                '2_line':lines,
                '3_bbox':bboxs})



    
    
