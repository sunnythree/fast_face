import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class CnnBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, active=nn.ReLU()):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.bn = nn.BatchNorm2d(planes)
        self.active = active

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.active(out)
        return out



class CnnAlign(nn.Module):
    def __init__(self):
        super(CnnAlign, self).__init__()
        self.cnn1 = CnnBlock(3, 16)
        self.cnn2 = CnnBlock(16, 32)
        self.cnn3 = CnnBlock(32, 64)
        self.cnn4 = CnnBlock(64, 128)
        self.cnn5 = CnnBlock(128, 128, kernel_size=1, padding=0)
        self.cnn6 = CnnBlock(128, 10, kernel_size=1, padding=0, active=nn.Tanh())
        self.max_pool = nn.MaxPool2d(2)
        self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.cnn1(x)          # 64
        x = self.max_pool(x)     # 32
        x = self.cnn2(x)         # 32
        x = self.max_pool(x)     # 16
        x = self.cnn3(x)         # 16
        x = self.max_pool(x)     # 8
        x = self.cnn4(x)         # 8
        x = self.max_pool(x)     # 4
        x = self.cnn5(x)         # 4
        x = self.mean_pool(x)    # 1
        x = self.cnn6(x)       # 1
        return torch.flatten(x, 1)


def test_model():
    net = CnnAlign()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())
if __name__=='__main__':
    test_model()