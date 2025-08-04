import torch
import torch.nn as nn
import torch.nn.functional as F
from src.entities.tnet import TNet

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.input_transform = TNet(3)
        self.feature_transform = TNet(64)

        #1d convolutional layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        #Batch normalizers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        B, N, _ = x.size()
        x = x.transpose(2,1)
        T = self.input_transform(x)
        x = torch.bmm(T, x)

        x = F.relu(self.bn1(self.conv1(x)))

        T_feat = self.feature_transform(x)
        x = torch.bmm(T_feat, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))

        x = self.fc3(x)
        return x




