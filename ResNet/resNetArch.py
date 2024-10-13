import torch
import torch.nn as nn

class convLayerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.convLayer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bnLayer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bnLayer(self.convLayer(x))
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()

        resChans = in_channels // 4
        stride = 1

        self.proj = in_channels != out_channels
        if self.proj: 
            self.skip_connection_convLayer = convLayerBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2
            resChans = in_channels // 2
        
        if first:
            self.skip_connection_convLayer = convLayerBlock(in_channels, out_channels, 1, 1, 0)
            stride = 1
            resChans = in_channels

        self.conv1 = convLayerBlock(in_channels, resChans, 1, 1, 0)
        self.conv2 = convLayerBlock(resChans, resChans, 3, stride, 1)
        self.conv3 = convLayerBlock(resChans, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))

        if self.proj:
            x = self.skip_connection_convLayer(x)

        output = self.relu(torch.add(f, x)) # Skip Connection's output is merged with the output flowing from the main network, here!!

        return output
    
class ResNet(nn.Module):
    def __init__(self, config: int, in_channels=3, classes=1000):
        super().__init__()
        configurations = {
            50 : [3, 4, 6, 3],
            101 : [3, 4, 23, 3],
            152 : [3, 8, 36, 3]
        }

        blocksNumber = configurations[config]
        out_features = [256, 512, 1024, 2048]

        self.blocks = nn.ModuleList([ResNetBlock(64, 256, True)])

        self.conv1 = convLayerBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        for i in range(len(blocksNumber)):
            if i > 0: self.blocks.append(ResNetBlock(out_features[i-1], out_features[i]))
            for _ in range(blocksNumber[i] - 1):
                self.blocks.append(ResNetBlock(out_features[i], out_features[i]))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, classes)
        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


config_name = 50 
resnet50 = ResNet(config_name)
image = torch.rand(1, 3, 224, 224)
print(resnet50(image).shape)
    