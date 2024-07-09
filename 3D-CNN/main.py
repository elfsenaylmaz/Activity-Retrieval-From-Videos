from video_dataset import VideoDataset
from CNN_3D import CNN3D
from train import train
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim;

class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

transform = transforms.Compose([
    NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = VideoDataset(root_dir='/content/drive/MyDrive/Kinetics400_Best/Videos/train', transform=transform)
valid_dataset = VideoDataset(root_dir='/content/drive/MyDrive/Kinetics400_Best/Videos/validate', transform=transform)


train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_data_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=0)

model = CNN3D(num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

train(model, train_data_loader, valid_data_loader, criterion, optimizer, num_epochs=5)