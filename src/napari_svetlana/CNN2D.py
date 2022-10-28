import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Module, BatchNorm2d, Softmax


class CNN2D(Module):
    def __init__(self, labels_number, channels_nb, depth, kersize):
        super(CNN2D, self).__init__()

        nconv = 16
        npool = 2
        layers_list = []

        for i in range(depth):
            if i == 0:
                # Defining a 2D convolution layer
                layers_list.append(Conv2d(channels_nb, nconv, kernel_size=(kersize, kersize), stride=(1, 1),
                                          padding=(1, 1)))
                layers_list.append(ReLU(inplace=True))
                layers_list.append(BatchNorm2d(nconv))
                layers_list.append(MaxPool2d(kernel_size=npool, stride=npool))
            else:
                # Defining a 2D convolution layer
                layers_list.append(Conv2d(nconv, nconv, kernel_size=(kersize, kersize), stride=(1, 1),
                                          padding=(1, 1)))
                layers_list.append(ReLU(inplace=True))
                layers_list.append(BatchNorm2d(nconv))
                layers_list.append(MaxPool2d(kernel_size=npool, stride=npool))

        self.cnn_layers = Sequential(*layers_list)
        self.avg_pool = Sequential(AdaptiveAvgPool2d(output_size=(1, 1)))
        self.fc = Sequential(Linear(nconv, labels_number))
        self.softmax = Softmax()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
