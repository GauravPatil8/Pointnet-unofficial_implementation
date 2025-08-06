import os
import glob
import numpy as np
import torch
from src.utils.data import read_off, get_classes
from torch.utils.data import Dataset

class ModelNet(Dataset):
    def __init__(self, data_dir, split='train', num_points=2048):
        self.num_points = num_points
        self.classes = get_classes(data_dir)
        self.classes_to_index = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.files = []


        for cls in self.classes:
            split_dir = os.path.join(data_dir, cls, split)
            for f in glob.glob(os.path.join(split_dir,'*.off')):
                self.files.append((f, self.classes_to_index[cls]))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path, label = self.files[index]
        point_cloud = read_off(file_path, self.num_points)

        if point_cloud.shape[0] > self.num_points:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)

        point_cloud = point_cloud[choice]
        point_cloud = point_cloud - np.mean(point_cloud, axis=0) # center at origin
        point_cloud = point_cloud/ np.max(np.linalg.norm(point_cloud, axis=1)) # normalize scale

        return torch.from_numpy(point_cloud).float(), torch.tensor(label, dtype=torch.long)