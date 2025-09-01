import pickle
import numpy as np

save_dt = {
    
    'pcd2_#888888': np.random.rand(5, 100, 3),  # Point cloud
    'line1_#123456': np.random.rand(5, 100, 2, 3),  # Line segments
    'bbox1_#123456': np.array([
        [[0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],]
        ]),  # Bounding box
    'mesh': {
        'vertex': np.random.rand(233, 6), # or (N, 6), (N, 7)
        'face':   np.random.randint(0, 233, size=(514, 3)),
    },
    'pcd1_#00FF0055': np.random.rand(100, 3),  # Point cloud
}

with open("example.pkl", 'wb') as f:
    pickle.dump(save_dt, f)