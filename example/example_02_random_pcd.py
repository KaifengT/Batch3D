
import numpy as np
import os, sys
import pickle
import time
import struct




def loop():
    
    '''
    format:
        pcd:
            vertex: (N, 3) x, y, z
            color:  (N, 3) r, g, b | (N, 4) r, g, b, a
            color: str: 'HHHHHH' | 'HHHHHHHH'
            
        line:
            (N, 2, 3) ((x1, y1, z1), (x2, y2, z2))
            color:  (N, 3) r, g, b | (N, 4) r, g, b, a
            color: str: 'HHHHHH' | 'HHHHHHHH'
            
        boundingbox:
            (8, 3)  ((x1, y1, z1), ... ,(x8, y8, z8))
            color:  (24, 3) r, g, b | (24, 4) r, g, b, a
            color: str: 'HHHHHH' | 'HHHHHHHH'
    
    '''

    pcd = np.random.rand(20000, 3) 
    pcd2 = np.random.rand(20000, 3) 
    
    lines = np.random.rand(100, 2, 3) 
    bboxs = np.random.rand(8, 3)
    lcolor = np.random.rand(200, 3)
    
    print('viewer', {'ID':1, 'objType':'p', 'vertex':pcd2, 'color':'FF3456', 'size':8.0})
    
    # print('pointcloud', {'ID':0, 'vertex':pcd, 'color':'FF3456'})
    # print('pointcloud', {'ID':1, 'vertex':pcd2 - 1, 'color':pcd2})
    # print('line',       {'ID':2, 'vertex':lines + 1, 'color':lcolor})
    # print('boundingbox',{'ID':3, 'vertex':bboxs + 2, 'color':'ff0000'})
    
    # Note: This will delete the object from sence with ID 3.
    # print('boundingbox',{'ID':3, 'vertex':None, })
    
    time.sleep(0.01)
    
    
