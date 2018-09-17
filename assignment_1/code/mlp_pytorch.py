"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# EXT
from torch import nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, custom_init=True):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """
        super().__init__()

        if len(n_hidden) == 0:
            self.layers = [self._create_linear(n_inputs, n_classes, custom_init)]
        else:
            self.layers = [self._create_linear(n_inputs, n_hidden[0], custom_init)]

            for layer_index, layer_size in list(enumerate(n_hidden + [n_classes]))[1:]:
                self.layers.append(nn.ReLU())
                #self.layers.append(nn.Dropout(p=0.2))
                self.layers.append(self._create_linear(n_hidden[layer_index - 1], layer_size, custom_init))

        self.model = nn.Sequential(*self.layers)

    @staticmethod
    def _create_linear(*dims, custom_init=True):
        """
        Initialize weights and bias similar to the Numpy MLP.
        """
        linear = nn.Linear(*dims)

        if custom_init:
            linear.weight.data.normal_(mean=0, std=0.0001)
            linear.bias.data.zero_()

        return linear

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        return self.model(x)
