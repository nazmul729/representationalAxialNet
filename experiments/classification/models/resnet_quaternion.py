'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quaternionconv.quaternion_layers import QuaternionConv

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = QuaternionConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = QuaternionConv(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuaternionConv(planes, self.expansion*planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        print("After Conv1: ",out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        print("After Conv2: ",out.shape)
        out = self.bn3(self.conv3(out))
        print("After Conv1: ",out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    expansion = 4
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.num_classes = num_classes
        self.conv1 = QuaternionConv(4, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, num_blocks[3], stride=2)
        self.linear = nn.Linear(1024*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        num = self.num_classes
        #print(num)
        out = F.relu(self.bn1(self.conv1(x)))
        
        print("After Conv1/stem: ",out.shape)
        out = self.maxpool(out)
        
        print("After max-pool: ",out.shape)
        out = self.layer1(out)
        print("After Layer1: ",out.shape)
        
        out = self.layer2(out)
        print("After Layer2: ",out.shape)
        
        out = self.layer3(out)
        print("After Layer3: ",out.shape)
        
        out = self.layer4(out)
        print("After Layer4: ",out.shape)
        
        out = F.avg_pool2d(out, 4)
        #print(out.shape)
        
        out = out.view(out.size(0), -1)
        print("Before linear: ",out.shape)
        
        out = self.linear(out)
        #print(out.shape)
        
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet26(num_classes):
    return ResNet(BasicBlock, [2, 4, 4, 2], num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == '__main__':
    net = ResNet18(1000)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    y = net(torch.randn(128, 3, 256, 256))
    print(y.size())
