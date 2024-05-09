import numpy as np
from scipy.spatial.transform import Rotation as R
from habitat_to_data import Dataset

ts = []
trainset = Dataset(
            training=True,
            save_fp="./train/",
            device="cuda",
        )
num_ims = 20
with np.load('data.npz') as data:
    a = data['poses']
    b = data['rgbd']
    for i in range(num_ims):
            pose = data['poses'][i].copy()

            T = np.eye(4)
            T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
            T[:3, 3] = pose[:3]
            ts.append(T)

    trainset.update_data(data['rgbd'][:num_ims, :, :, :3], data['rgbd'][:num_ims, :, :, 3], np.array(ts))

    trainset.save()

