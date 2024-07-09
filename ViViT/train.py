import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from video_dataset import VideoDataset  # Video veri seti modülünüzü ekleyin
from video_dataset import NormalizeVideo
from ViVit import VideoVisionTransformer
from tqdm import tqdm
from torchvision import transforms
import random
import torchvision
import numpy as np

batch_size = 8
num_epochs = 10
learning_rate = 0.00001

model = VideoVisionTransformer().cuda()
#model.load_state_dict(torch.load("/content/drive/MyDrive/Kinetics400/Model/model15(100).pt"))

tfs = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.),
#             reshape into (C, T, H, W) for easier convolutions
            torchvision.transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),
#             rescale to the most common size
            torchvision.transforms.Lambda(lambda x: nn.functional.interpolate(x, (224, 224))),
])

transform = transforms.Compose([
    NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = VideoDataset(root_dir='/content/drive/MyDrive/Kinetics400_Best/Videos/train', transform=transform)
valid_dataset = VideoDataset(root_dir='/content/drive/MyDrive/Kinetics400_Best/Videos/validate', transform=transform)


train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_data_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=0)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=0)

weight_decay = 1e-4
criterion = torch.nn.CrossEntropyLoss().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)


def train(epoch):
    model.train()
    train_loss = 0
    running_score = 0
    i = 0
    for (train_input, train_target) in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        try:
            train_input = train_input.cuda()
            train_input = train_input.to(torch.float32)
            train_target = train_target.cuda()
            optimizer.zero_grad()   # zero the gradient buffers
            output = model(train_input)
            val, index_ = torch.max(output, axis=1)
            running_score += torch.sum(index_ == train_target.data).item()
            loss = criterion(output, train_target)
            
            if i % 50 == 0:
                print()
                print("mini_train_loss: ", loss.item())
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()    # Does the 
            i += 1
        except Exception as e:
            print("Train Error!")
            print(e)
    
        
    print("train_loss: ", train_loss / (i + 1))    
    epoch_score = running_score/len(train_loader.dataset)
    print("Train Accuracy: %", epoch_score * 100)
    if epoch % 1 == 0 and epoch != 0:
        torch.save(model.state_dict(), f"/content/drive/MyDrive/Models/ViVit/model_{epoch}_20Class_all.pth")

def evaluate(test_loss_min):
    model.eval()
    test_loss = 0
    running_score = 0.0
    test_i = 0
    total = 0
    try:
        for (test_input, test_target) in tqdm(valid_loader, desc='Testing', leave=False):
            test_input = test_input.cuda()
            test_target = test_target.cuda()
            output = model(test_input)
            loss = criterion(output, test_target)
            val, index_ = torch.max(output, axis=1)
            running_score += torch.sum(index_ == test_target.data).item()
            total += len(index_)
            test_loss += loss.item()
            test_i += 1
    except Exception as er:
        print("Test Error!")
        print(er)
    
    test_loss = test_loss / (test_i + 1)
    epoch_score = running_score/total
    print("test_loss: ", test_loss)
    print("Test Accuracy: %", epoch_score * 100)
    
    if test_loss < test_loss_min:
        # save model if validation loss has decreased
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min, test_loss))
        test_loss_min = test_loss
    return test_loss_min

test_loss_min = np.Inf
for epoch in range(num_epochs):
    print("EPOCH: ", epoch)
    print("________________________________")
    print()
    train(epoch)
    test_loss_min = evaluate(test_loss_min)
