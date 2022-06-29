import torch
from torch.nn import Sequential, Conv3d, BatchNorm3d, ReLU, MaxPool3d, AdaptiveAvgPool3d, Linear, Module


class CNN3D(Module):
    def __init__(self, labels_number, channels_nb):
        super(CNN3D, self).__init__()

        self.conv1 = Conv3d(channels_nb, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm3d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            BatchNorm3d(128),
            ReLU(inplace=True),
            MaxPool3d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv3d(128, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm3d(512),
            ReLU(inplace=True),
            MaxPool3d(kernel_size=2, stride=2),
        )
        self.avg_pool = Sequential(AdaptiveAvgPool3d(output_size=(1, 1, 1)))
        self.fc = Sequential(Linear(512, labels_number))

    # Defining the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.cnn_layers(x)
        #x = x.view(x.size(0), -1)
        x = self.avg_pool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x
