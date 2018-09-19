"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# EXT
from torch import nn

from operator import mul
from functools import reduce


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super().__init__()

        # Implement version of the VGG net
        # All the layers are pretty much pre-defined

        # 1. Block
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. Block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3. Block
        self.conv3_a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.batchnorm3_a = nn.BatchNorm2d(256)
        self.relu3_a = nn.ReLU()
        self.conv3_b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.batchnorm3_b = nn.BatchNorm2d(256)
        self.relu3_b = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4. Block
        self.conv4_a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.batchnorm4_a = nn.BatchNorm2d(512)
        self.relu4_a = nn.ReLU()
        self.conv4_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.batchnorm4_b = nn.BatchNorm2d(512)
        self.relu4_b = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 5. Block
        self.conv5_a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.batchnorm5_a = nn.BatchNorm2d(512)
        self.relu5_a = nn.ReLU()
        self.conv5_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.batchnorm5_b = nn.BatchNorm2d(512)
        self.relu5_b = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Output block
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(512, n_classes)

        self.features = nn.Sequential(
            self.conv1, self.batchnorm1, self.relu1, self.maxpool1,
            self.conv2, self.batchnorm2, self.relu2, self.maxpool2,
            self.conv3_a, self.batchnorm3_a, self.relu3_a, self.conv3_b, self.batchnorm3_b, self.relu3_b, self.maxpool3,
            self.conv4_a, self.batchnorm4_a, self.relu4_a, self.conv4_b, self.batchnorm4_b, self.relu4_b, self.maxpool4,
            self.conv5_a, self.batchnorm5_a, self.relu5_a, self.conv5_b, self.batchnorm5_b, self.relu5_b, self.maxpool5,
            self.avgpool
        )
        self.classifier = self.linear

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        feat = self.features(x)
        feat = feat.squeeze()
        out = self.classifier(feat)

        return out
