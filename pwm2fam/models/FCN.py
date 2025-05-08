import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F

class FCN(nn.Module):
    
    def __init__(self, out_channels):
        super(FCN, self).__init__()
        
        # conv1
        self.conv1 = Conv1d(4, 16, kernel_size=3, padding=1)

        # conv2
        self.conv2 = Conv1d(16, 16, kernel_size = 5, padding=2)      

        # conv3
        self.conv3 = Conv1d(16, out_channels, kernel_size = 7, padding=3)
        
        # fully connected
        self.fc = Conv1d(out_channels, out_channels, kernel_size=1)

    def strand_forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
       
        x = self.fc(x)
       
        x, _ = torch.max(x, dim=2)
        return x

    def forward(self, x):
        logits1 = self.strand_forward(x)
        x_flipped = torch.flip(x, dims=[1,2])
        logits2 = self.strand_forward(x_flipped)
        
        return (logits1 + logits2)/2 