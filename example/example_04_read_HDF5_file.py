import numpy as np
import h5py



# 方式1
# 保存到 HDF5 文件
output_file = "point_cloud.h5"
n = 1000
with h5py.File(output_file, "w") as h5f:
    h5f.create_dataset("point_cloud_1", data=np.random.rand(n, 6))
    h5f.create_dataset("point_cloud_2", data=np.random.rand(n, 6))
    h5f.create_dataset("point_cloud_3", data=np.random.rand(n, 6))

print(f"Point cloud saved to {output_file}")

# 方式1 使用Group
n = 5
output_file = "point_cloud_group.h5"
# 需要将track_order设为True，以保证读取时的顺序与写入时一致
with h5py.File(output_file, "w", track_order=True) as h5f:
    for i in range(10):
        group = h5f.create_group(f"point_cloud_{i+1:0>4d}")
        group.create_dataset("data_1", data=np.random.rand(n, 6))
        group.create_dataset("line_2", data=np.random.rand(n, 2, 3))
        group.create_dataset("bbox_3", data=np.random.rand(n, 8, 3))
print(f"Point cloud saved to {output_file}")