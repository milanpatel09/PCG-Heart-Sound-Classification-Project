import torch
from torch.utils.data import Dataset 
import numpy as np
import torchvision.transforms as T

class PCGDataset(Dataset):
    def __init__(self, feature_path, label_path):
        """
        Loads features and labels.
        1. Normalizes (Z-Score).
        2. Resizes to 224x224 (as per Bao et al. paper).
        3. Converts to 3 Channels (RGB) for ResNet.
        """
        self.y = np.load(label_path)
        # mmap_mode='r' keeps memory usage low
        self.X = np.load(feature_path, mmap_mode='r')
        
        # Define the Resize transform (Standard ImageNet Size)
        # antialias=True prevents artifacts when shrinking large images (like CWT)
        self.resize = T.Resize((224, 224), antialias=True)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # 1. Load data
        data = np.array(self.X[idx]) 
        label = self.y[idx]
        
        # 2. Z-Score Normalization
        mean = data.mean()
        std = data.std()
        # 1e-6 prevents division by zero
        data_norm = (data - mean) / (std + 1e-6)
            
        # 3. Create Tensor and Add Channel Dimension: (1, H, W)
        data_tensor = torch.tensor(data_norm).float().unsqueeze(0)
        
        # 4. Resize to 224x224
        # This squashes the time/freq dimensions to fit the square
        data_resized = self.resize(data_tensor)
        
        # 5. Convert to 3 Channels (RGB)
        # We replicate the grayscale image 3 times.
        # Shape becomes: (3, 224, 224)
        data_rgb = data_resized.repeat(3, 1, 1)
        
        return data_rgb, torch.tensor(label).long()