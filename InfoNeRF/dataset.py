import jittor as jt
from jittor.dataset import Dataset


class RayDataset(Dataset):
    def __init__(self, ray_data, batch_size, shuffle):
        super(RayDataset, self).__init__()

        self.rayData = ray_data
        self.length = ray_data.shape[0]

        self.set_attrs(batch_size=batch_size,
                       total_len=self.length, shuffle=shuffle)

    def __getitem__(self, index):
        return jt.float32(self.rayData[index])


class RayPoseDataset(Dataset):
    def __init__(self, ray_data, ray_pose, batch_size, shuffle):
        super(RayPoseDataset, self).__init__()

        self.rayData = ray_data
        self.rayPose = ray_pose
        self.length = ray_data.shape[0]

        self.set_attrs(batch_size=batch_size,
                       total_len=self.length, shuffle=shuffle)

    def __getitem__(self, index):
        return jt.float32(self.rayData[index]), jt.float32(self.rayPose[index])
