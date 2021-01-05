import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_dims=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Student(nn.Module):
    def __init__(self, temperature):
        super(Student, self).__init__()
        self.temperature = temperature
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.mp1(F.relu(self.bn1(self.cnn1(x))))
        x = self.mp2(F.relu(self.bn2(self.cnn2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)
        soft_target = F.log_softmax(x / self.temperature, dim=1)
        hard_target = F.log_softmax(x, dim=1)

        return soft_target, hard_target


# Softmax with temperature
# -- Adapted from PyTorch Softmax layer
# -- See: https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Softmax
class SoftmaxT(nn.Module):
    def __init__(self, temperature, dim=1) -> None:
        super(SoftmaxT, self).__init__()
        self.temperature = temperature
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, in_data):
        return F.log_softmax(in_data / self.temperature, self.dim)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)


def create_student(device, temperature):
    return Student(temperature).to(device)


def create_teacher(device, temperature, teacher_state_dict_path=None, in_dims=3):
    teacher_model = ResNet(BasicBlock, [2, 2, 2, 2], in_dims).to(device)

    if teacher_state_dict_path is not None:
        teacher_model.load_state_dict(torch.load(teacher_state_dict_path, map_location=device))

    return torch.nn.Sequential(
        teacher_model,
        SoftmaxT(temperature)
    ).to(device)
