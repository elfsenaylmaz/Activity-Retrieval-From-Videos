import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from video_dataset import VideoDataset  # Video veri seti modülünüzü ekleyin
from ViVit import VideoVisionTransformer
from tqdm import tqdm
import random
import torchvision
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sns

class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

transform = transforms.Compose([
    NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


batch_size = 4

model = VideoVisionTransformer().cuda()
model.load_state_dict(torch.load("/content/drive/MyDrive/Models/ViVit/model_9_20Class_all.pth"))

tfs = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.),
#             reshape into (C, T, H, W) for easier convolutions
            torchvision.transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),
#             rescale to the most common size
            torchvision.transforms.Lambda(lambda x: nn.functional.interpolate(x, (224, 224))),
])

test_dataset = VideoDataset(root_dir='/content/drive/MyDrive/Kinetics400_Best/Videos/test', transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

criterion = torch.nn.CrossEntropyLoss().to("cuda")

all_outputs = []
all_targets = []

model.eval()
test_loss = 0
running_score = 0.0
test_i = 0
total = 0
try:
    for (test_input, test_target) in tqdm(test_loader, desc='Testing', leave=False):


        test_input = test_input.cuda()
        test_target = test_target.cuda()

        output = model(test_input)
        loss = criterion(output, test_target)
        val, index_ = torch.max(output, axis=1)
        running_score += torch.sum(index_ == test_target.data).item()
        test_loss += loss.item()
        test_i += 1
        all_outputs.extend(index_.tolist())
        all_targets.extend(test_target.tolist())
except Exception as e:
    print("Total ", total)
    print("Score ", running_score)
    print("Error")
    print(e)


test_loss = test_loss / (test_i + 1)
epoch_score = running_score/len(test_loader.dataset)
print("test_loss: ", test_loss)
print("Test Accuracy: ", epoch_score, "%")
cm = confusion_matrix(all_targets, all_outputs)
print("Confusion Matrix:")
print(cm)

report = classification_report(all_targets, all_outputs, target_names=test_dataset.classes)
print("Classification Report:")
print(report)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen Değer")
plt.ylabel("Gerçek Değer")
plt.show()