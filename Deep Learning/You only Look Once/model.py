import torch
import torch.nn as nn

class Yolo(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 64):
        super(Yolo, self).__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 0))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(out_channels, out_channels*3, kernel_size = 3))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(out_channels*3, out_channels*2, kernel_size = 1))
        layers.append(nn.Conv2d(out_channels*2, out_channels*4, kernel_size = 3))
        layers.append(nn.Conv2d(out_channels*4, out_channels*4, kernel_size = 1))
        layers.append(nn.Conv2d(out_channels*4, out_channels*8, kernel_size = 1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(2):
            layers.append(nn.Conv2d(out_channels*8, out_channels*4, kernel_size = 1))
            layers.append(nn.Conv2d(out_channels*4, out_channels*8, kernel_size = 3))
            layers.append(nn.Conv2d(out_channels*8, out_channels*4, kernel_size = 1))
            layers.append(nn.Conv2d(out_channels*4, out_channels*8, kernel_size = 3))

        layers.append(nn.Conv2d(out_channels*8, out_channels*8, kernel_size = 1))
        layers.append(nn.Conv2d(out_channels*8, out_channels*16, kernel_size = 3))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(out_channels*16, out_channels*8, kernel_size = 1))
        layers.append(nn.Conv2d(out_channels*8, out_channels*16, kernel_size = 3))
        layers.append(nn.Conv2d(out_channels*16, out_channels*8, kernel_size = 1))
        layers.append(nn.Conv2d(out_channels*8, out_channels*16, kernel_size = 3))

        layers.append(nn.Conv2d(out_channels*16, out_channels*16, kernel_size = 3))
        layers.append(nn.Conv2d(out_channels*16, out_channels*16, kernel_size = 3, stride = 2))
        layers.append(nn.Conv2d(out_channels*16, out_channels*16, kernel_size = 3))
        layers.append(nn.Conv2d(out_channels*16, out_channels*16, kernel_size = 3))

        self.fc1 = nn.Linear(7*7*1024, 4096)
        self.fc2 = nn.Linear(4096, 1470)
        
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        out = self.main(input)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
