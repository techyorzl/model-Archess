#In this script, I implemented VGG-16 proposed by Visual Geometry Group, by just using PyTorch. It is a demo model and shows the architecture of VGG-16!!

import torch
import torch.nn as nn

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=7*7*512, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):

        x = self.maxpool1(torch.relu(self.conv1_2(torch.relu(self.conv1_1(x)))))

        x = self.maxpool2(torch.relu(self.conv2_2(torch.relu(self.conv2_1(x)))))

        x = self.maxpool3(torch.relu(self.conv3_3(torch.relu(self.conv3_2(torch.relu(self.conv3_1(x)))))))

        x = self.maxpool4(torch.relu(self.conv4_3(torch.relu(self.conv4_2(torch.relu(self.conv4_1(x)))))))

        x = self.maxpool5(torch.relu(self.conv5_3(torch.relu(self.conv5_2(torch.relu(self.conv5_1(x)))))))

        x = x.view(-1, 7*7*512)

        x = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

        return x

img = torch.rand(224, 224, 3)
img = img.permute(2, 0, 1).unsqueeze(0)    
model = vgg16()
model.eval()
output = model(img)
print(output.shape)
print(output)
