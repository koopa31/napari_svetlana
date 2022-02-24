import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Module, BatchNorm2d


class CNN2D(Module):
    def __init__(self, labels_number, channels_nb):
        super(CNN2D, self).__init__()

        kersize = 3
        nconv = 16
        npool = 2
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(channels_nb, nconv, kernel_size=(kersize, kersize), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            BatchNorm2d(nconv),
            #Conv2d(nconv, nconv, kernel_size=(kersize, kersize), stride=(1, 1), padding=(1, 1)),
            #ReLU(inplace=True),
            #BatchNorm2d(nconv),
            MaxPool2d(kernel_size=npool, stride=npool),
            # Defining a 2D convolution layer
            Conv2d(nconv, nconv, kernel_size=(kersize, kersize), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            BatchNorm2d(nconv),
            MaxPool2d(kernel_size=npool, stride=npool),
            # Defining a 2D convolution layer
            #Conv2d(nconv, nconv, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            #ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=2),
        )
        self.avg_pool = Sequential(AdaptiveAvgPool2d(output_size=(1, 1)))
        self.fc = Sequential(Linear(nconv, labels_number + 1))

    # Defining the forward pass
    def forward(self, x):

        x = self.cnn_layers(x)
        #x = x.view(x.size(0), -1)
        x = self.avg_pool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x
