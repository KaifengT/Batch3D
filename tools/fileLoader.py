import numpy as np
import os, sys
from plyfile import PlyData

class BaseLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        self.data = self.load()
        
        
    def load(self):
        ...


class PLYLoader(BaseLoader):
    
    
    def load(self, ):
        def npmemmap_to_array(memmap:np.memmap):
            return np.array(memmap.tolist())

    # Read the PLY file
        plydata = PlyData.read(self.filepath)

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
        
        multiVertex = np.concatenate([v, c], axis=1)
        
        return {'vertex':multiVertex, }
