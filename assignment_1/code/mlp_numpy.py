"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import modules


class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
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
    self.input_layer = modules.LinearModule(n_inputs, n_classes if len(n_hidden) == 0 else n_hidden[0])
    self.hidden_layers = []

    layer_sizes = n_hidden + [n_classes]
    for layer, next_hidden in list(enumerate(layer_sizes))[1:]:
        self.hidden_layers.append(modules.LinearModule(layer_sizes[layer - 1], next_hidden))

    self.output_layer = modules.SoftMaxModule()

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    out = self.input_layer.forward(x)

    for hidden_layer in self.hidden_layers:
        out = hidden_layer.forward(out)

    out = self.output_layer.forward(out)

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    """
    dout = self.output_layer.backward(dout)

    for hidden_layer in self.hidden_layers:
        dout = hidden_layer.backward(dout)

    dout = self.input_layer.backward(dout)

    return dout
