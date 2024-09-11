import torch
import torch.nn as nn

class alexNet(nn.Module):

    def __init__(self):
        super(alexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.max4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)


    def forward(self, x):
        x = self.max1(torch.relu(self.conv1(x)))

        x = self.max2(torch.relu(self.conv2(x)))

        x = self.max3(torch.relu(self.conv3(x)))

        x = torch.relu(self.conv4(x))

        x = self.max4(torch.relu(self.conv5(x)))

        x = x.view(-1, 6*6*256)

        x = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

        return x
    
img = torch.rand(224, 224, 3)
img = img.permute(2, 0, 1).unsqueeze(0)    
model = alexNet()
model.eval()
output = model(img)
print(output.shape)
print(output)
