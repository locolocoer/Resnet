import torch
import torch.nn as nn
import torch.nn.functional as  F


class lambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(lambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.BN1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if inplanes != planes or stride != 1:
            if option == 'A':
                self.shortcut = lambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), mode="constant",
                                    value=0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.BN1(self.conv1(x)))
        out = self.BN2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    pass


class resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(resnet, self).__init__()
        self.in_planes = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[1], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.GAvaPool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(64, num_classes)

    def _make_layer(self, block, plane, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)

        for stride in strides:
            layers.append(block(self.in_planes, plane, stride))
            self.in_planes = plane * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.GAvaPool(x)
        x = x.view(x.size(0), -1)
        return self.linear1(x)


if __name__ == "__main__":
    #block = BasicBlock(16, 32, 2)
    x = torch.ones((1, 3, 32, 32))
    #y = block(x)
    #Resnet20 = resnet(BasicBlock, [3, 3, 3])
    #y=Resnet20(x)
    #print(y)

    gap=nn.AdaptiveAvgPool2d((1,1))
    print(gap(x).view(1,-1).size())
