"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample
    """
    self.params = {'weight': np.random.normal(0, 0.0001, (in_features, out_features)), 'bias': np.zeros(out_features)}
    self.grads = {'weight': np.zeros((in_features, out_features)), 'bias': np.zeros(out_features)}

    # Stored values for gradients
    self.x = None
    self.out_features = out_features

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """
    self.x = x
    out = self.params["weight"] @ x + self.params["bias"]

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """
    self.grads["weights"] = self.x
    self.grads["bias"] = np.ones(self.out_features)
    dx = self.grads["weights"] * dout

    return dx


class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """
    out = np.maximum(np.zeros(x.shape[0]), x)

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    """
    dx = dout

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def __init__(self):
    # Stored values for gradients
    self.x = None
    self.dim = None

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module                                                        #
    """
    self.x = x
    self.dim = x.shape[0]
    out = self.softmax(x)

    return out

  @staticmethod
  def softmax(x):
    z = x.max()
    out = np.exp(x - z) / np.exp(x - z).sum()
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    """

    out = self.softmax(self.x)
    gradients = - out.T @ out
    # Add x_i^(N) where i = j
    gradients += np.diag(out)
    dx = dout * gradients

    return dx


class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    """
    out = - np.dot(np.log(x), y)

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    """
    dx = - y / x

    return dx
