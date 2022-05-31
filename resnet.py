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
        if inplanes != planes:
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


if __name__ == "__main__":
    block = BasicBlock(16, 32, 2)
    x = torch.ones((1, 16, 24, 24))
    y = block(x)

    print(y)
