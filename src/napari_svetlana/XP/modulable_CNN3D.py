import torch
from torch.nn import Sequential, Conv3d, BatchNorm3d, ReLU, MaxPool3d, AdaptiveAvgPool3d, Linear, Module, Softmax


class CNN3D(Module):
    def __init__(self, labels_number, channels_nb, width, depth):
        super(CNN3D, self).__init__()
        self.width = width
        self.pool = 2
        layers_list = []

        for i in range(depth):
            if i == 0:
                layers_list.append(Conv3d(channels_nb, self.width, kernel_size=7, stride=2, padding=3, bias=False))
                layers_list.append(ReLU(inplace=True))
                layers_list.append(BatchNorm3d(self.width))
                layers_list.append(MaxPool3d(kernel_size=2, stride=self.pool, padding=1))
            else:
                layers_list.append(Conv3d(self.width * i, self.width * 2 * i, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                          padding=(1, 1, 1)))
                layers_list.append(ReLU(inplace=True))
                layers_list.append(BatchNorm3d(self.width * 2 * i))
                layers_list.append(MaxPool3d(kernel_size=2, stride=self.pool))

        self.cnn_layers = Sequential(*layers_list)
        self.avg_pool = Sequential(AdaptiveAvgPool3d(output_size=(1, 1, 1)))
        self.fc = Sequential(Linear(self.width * 2 * i, labels_number))
        self.softmax = Softmax()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
