import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Module, BatchNorm2d
from torch import nn


"""class CNN2D(Module):
    def __init__(self, labels_number, channels_nb, depth, kersize):
        super(CNN2D, self).__init__()

        depth = 1
        kersize = 15
        nconv = 2
        npool = 2
        first_later = []
        layers_list = []
        mask_conv = []

        for i in range(depth):
            if i == 0:
                # Defining a 2D convolution layer
                first_later.append(Conv2d(3, nconv, kernel_size=(kersize, kersize), stride=(1, 1),
                                          padding=(1, 1)))
                first_later.append(ReLU(inplace=True))
                first_later.append(BatchNorm2d(nconv))
                first_later.append(MaxPool2d(kernel_size=npool, stride=npool))
            else:
                # Defining a 2D convolution layer
                layers_list.append(Conv2d(nconv, nconv, kernel_size=(kersize, kersize), stride=(1, 1),
                                          padding=(1, 1)))
                layers_list.append(ReLU(inplace=True))
                layers_list.append(BatchNorm2d(nconv))
                layers_list.append(MaxPool2d(kernel_size=npool, stride=npool))

        mask_conv.append(Conv2d(1, nconv, kernel_size=(kersize, kersize), stride=(1, 1), padding=(1, 1)))
        mask_conv.append(ReLU(inplace=True))
        mask_conv.append(BatchNorm2d(nconv))
        mask_conv.append(MaxPool2d(kernel_size=npool, stride=npool))
        self.mask_conv = Sequential(*mask_conv)
        self.first_layer = Sequential(*first_later)
        self.cnn_layers = Sequential(*layers_list)
        self.avg_pool = Sequential(AdaptiveAvgPool2d(output_size=(1, 1)))
        self.fc = Sequential(Linear(nconv, labels_number + 1))

    # Defining the forward pass
    def forward(self, x):

        img = x[:, :3, :, :]
        mask = x[:, 3, :, :]
        x = self.first_layer(img)
        #mask = self.mask_conv(mask[:, None, :, :])
        #x = torch.mul(img, mask)
        x = self.cnn_layers(x)
        x = self.avg_pool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x

"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CNN2D(nn.Module):

    def __init__(self, labels_number, channels_nb, depth, kersize):
        super().__init__()
        layers = [1, 1]
        block = BasicBlock

        self.inplanes = 8

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_mask = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                                   bias=True)

        self.layer1 = self._make_layer(block, 4, layers[0])
        self.layer2 = self._make_layer(block, 8, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, labels_number + 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        img = x[:, :3, :, :]
        mask = x[:, 3, :, :]

        img = self.conv1(img)  # 224x224
        img = self.bn1(img)
        img = self.relu(img)
        img = self.maxpool(img)  # 112x112

        mask = self.conv_mask(mask[:, None, :, :])  # 224x224

        x = torch.add(img, mask)

        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        #x = self.layer3(x)  # 14x14
        #x = self.layer4(x)  # 7x7

        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)  # remove 1 X 1 grid and make vector of tensor shape
        x = self.fc(x)

        return x
