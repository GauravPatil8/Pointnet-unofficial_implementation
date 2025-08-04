import os
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from src.utils.common import read_yaml
from torch.utils.data import DataLoader
from src.entities.modelnet import ModelNet
from src.entities.pointnet import PointNet
from src.constants import CONFIG_PATH, PROJECT_ROOT


class Trainer:
    def __init__(self, epoch = 20):
        self.epoch = epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = read_yaml(CONFIG_PATH)
        self.data_dir = os.path.join(PROJECT_ROOT, self.config["data_ingestion"]["extract_path"])

        self.train_dataset = ModelNet(data_dir=self.data_dir, split="train")
        self.test_dataset = ModelNet(data_dir=self.data_dir, split="test")

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=True, num_workers=2)

        self.model = PointNet(num_classes = 10).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def __str__(self):
        return f"Running Model Training Stage"
    
    def train(self):
        self.model.train()
        total_loss, total_correct = 0.0, 0
        for data, label in tqdm(self.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer().step()
            total_loss+= loss.item()
            pred = out.argmax(dim = 1)
            total_correct += (pred == label).sum().item()
        acc = total_correct / len(self.train_loader)
        return total_loss / len(self.train_loader), acc
    
    def test(self):
        self.model.eval()
        total_loss, total_correct = 0.0, 0
        with torch.no_grad():
            for data, label in self.test_loader:
                data, label = data.to(self.device), label.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, label)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                total_correct += (pred == label).sum().item()
        acc = total_correct / len(self.test_loader.dataset)
        return total_loss / len(self.test_loader), acc
    
    def process(self):
        logger = logging.getLogger(__name__)
        for _ in range(self.epoch):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            logger.info(f"Epoch {self.epoch+1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
        try:
            torch.save(self.model.state_dict(), self.config["trainer"]["model_path"]) # only weights are stored
            logger.info(f"Trained model saved at {self.config["trainer"]["model_path"]}")

        except Exception as e:
            logger.error(f"An unexpected error occured: {e}")


