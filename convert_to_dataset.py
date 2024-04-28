import numpy as np
from scipy.spatial.transform import Rotation as R
from habitat_to_data import Dataset

ts = []
trainset = Dataset(
            training=True,
            save_fp="./train/",
            device="cuda",
        )
with np.load('data.npz') as data:
    a = data['poses']
    b = data['rgbd']
    for i in range(len(a)):
            pose = data['poses'][i].copy()

            T = np.eye(4)
            T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
            T[:3, 3] = pose[:3]
            ts.append(T)

    trainset.update_data(data['rgbd'][:, :, :, :3], data['rgbd'][:, :, :, 3], np.array(ts))

    trainset.save()

