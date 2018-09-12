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
        self.params = {'weight': np.random.normal(0, 0.0001, (in_features, out_features)),
                       'bias': np.zeros(out_features)}
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
        out = x @ self.params["weight"] + self.params["bias"]

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        self.grads["weight"] = self.x.T @ dout
        self.grads["bias"] = dout.mean(axis=0)
        dx = dout @ self.params["weight"].T

        return dx

    def update_parameters(self, learning_rate):
        self.params["weight"] -= learning_rate * self.grads["weight"]
        self.params["bias"] -= learning_rate * self.grads["bias"]


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        self.x = x
        out = np.maximum(np.zeros(x.shape), x)

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
        """
        dx = dout * (self.x > 0).astype(int)

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
        out = self.softmax(x)

        return out

    @staticmethod
    def softmax(x):
        z = x.max(axis=1, keepdims=True)
        z = np.repeat(z, x.shape[1], axis=1)
        out = np.exp(x - z) / np.exp(x - z).sum(axis=1, keepdims=True)
        test = out.sum(axis=1)  # TODO: Remove, just test if softmax adds up to 1
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
        # dx = np.empty((0, out.shape[1]))
        #
        # # TODO: Make function work without any loops whatsoever
        # for batch_instance, dout_ in zip(out, dout):
        #     # Perform batch-wise matrix multiplcation - numpy doesn't have bmm like PyTorch
        #     diag = np.diag(batch_instance)
        #     batch_instance = batch_instance[np.newaxis, ...]
        #     gradient = - batch_instance.T @ batch_instance
        #     # Add x_i^(N) where i = j
        #     gradient += diag
        #     gradient = (gradient @ dout_.T)[np.newaxis, ...]
        #     dx = np.concatenate((dx, gradient), axis=0)

        # Other approach with einsum
        # b: batch size
        # c: number of output classes

        # Create matrix of softmax values per batch instance b x c, b x c -> b x c x c
        gradient = - np.einsum("ij, ik -> ijk", out, out)

        # Create diagonals of softmax values per batch instance where i = j which will be added to previous matrices
        diagonals = np.apply_along_axis(np.diag, axis=1, arr=out)  # b x c x c where c x c are diagonal matrices
        gradient += diagonals
        dout = dout[..., np.newaxis]  # b x c -> b x c x 1

        # Create gradients using previous loss
        # b x c x c, b x c x 1 -> b x c x 1 -> b x c
        dx = np.einsum("ijk, ikl -> ijl", gradient, dout).squeeze(axis=2)

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
        epsilon = 1e-10
        x[x == 0] = epsilon
        out = - (np.log(x) * y).sum(axis=1)
        out = out.mean()  # Divide by batch size

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
        dx = dx / x.shape[0]  # Divide by batch size

        return dx
