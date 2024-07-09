import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video
import random
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=64):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.classes = self.classes[:20]

        print(self.classes)
        self.video_paths = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for i, video_name in enumerate(os.listdir(class_dir)):
                self.video_paths.append(os.path.join(class_dir, video_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        frame_idx = total_frames // 3
        frame_idx -= self.num_frames // 2

        if frame_idx < 0:
            frame_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
            if len(frames) >= self.num_frames:
                break

        cap.release()

        if len(frames) == 0:
            # If no frames are captured, create a dummy frame filled with zeros
            frames = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(self.num_frames)]
        elif len(frames) < 64:
            # Pad with the last frame if video is shorter than self.num_frames
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        elif len(frames) > 64:
            # Truncate frames if video is longer than self.num_frames
            frames = frames[:64]

        # Convert list of frames to numpy array and then to tensor
        frames = np.array(frames).astype(np.float32)  # (T, H, W, C)
        frames = torch.tensor(frames).permute(3, 0, 1, 2)  # (C, T, H, W)

        if self.transform:
            # Apply normalization to each frame separately
            frames = self.transform(frames / 255.0)

        return frames, label

class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std