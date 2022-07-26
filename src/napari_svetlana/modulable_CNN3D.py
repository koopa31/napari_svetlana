import torch
from torch.nn import Sequential, Conv3d, BatchNorm3d, ReLU, MaxPool3d, AdaptiveAvgPool3d, Linear, Module, Softmax


class CNN3D(Module):
    def __init__(self, labels_number, channels_nb):
        super(CNN3D, self).__init__()
        self.width = 2
        self.pool = 2

        self.conv1 = Conv3d(channels_nb, self.width, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = ReLU(inplace=True)
        self.bn1 = BatchNorm3d(self.width)
        self.maxpool = MaxPool3d(kernel_size=2, stride=self.pool, padding=1)

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv3d(self.width, self.width*2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ReLU(inplace=True),
            BatchNorm3d(self.width*2),
            MaxPool3d(kernel_size=2, stride=self.pool),
            # Defining another 2D convolution layer
            Conv3d(self.width*2, self.width*4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm3d(self.width*4),
            MaxPool3d(kernel_size=2, stride=self.pool),
        )
        self.avg_pool = Sequential(AdaptiveAvgPool3d(output_size=(1, 1, 1)))
        self.fc = Sequential(Linear(self.width*4, labels_number))
        self.softmax = Softmax()

    # Defining the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.cnn_layers(x)
        #x = x.view(x.size(0), -1)
        x = self.avg_pool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        #x = self.softmax(x)
        return x
