import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """
    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormAutograd, self).__init__()

        self.n_neurons = n_neurons
        self.epsilon = eps

        self.gamma = nn.Parameter(torch.ones(1, n_neurons))
        self.beta = nn.Parameter(torch.zeros(1, n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        batch_size = input.shape[0]

        # 1. Compute mean
        mu = input.mean(dim=0).unsqueeze(0)
        mu = mu.repeat(batch_size, 1)

        # 2. Compute variance
        sigma = input.var(dim=0, unbiased=False).unsqueeze(0)
        sigma = sigma.repeat(batch_size, 1)

        # 3. Normalize
        normalized = (input - mu) / torch.sqrt(sigma + self.epsilon)

        # 4. Scale and shift
        out = normalized * self.gamma.repeat(batch_size, 1)
        out += self.beta.repeat(batch_size, 1)

        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomBatchNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          input: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor
        """
        batch_size = input.shape[0]

        # 1. Compute mean
        mu = input.mean(dim=0).unsqueeze(0)
        mu = mu.repeat(batch_size, 1)

        # 2. Compute variance
        sigma = input.var(dim=0, unbiased=False).unsqueeze(0)
        sigma = sigma.repeat(batch_size, 1)

        # 3. Normalize
        normalized = (input - mu) / torch.sqrt(sigma + eps)

        # 4. Scale and shift
        out = normalized * gamma.repeat(batch_size, 1)
        out += beta.repeat(batch_size, 1)

        # Save for backward pass
        ctx.save_for_backward(gamma, beta, sigma, normalized)
        ctx.constant = eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrieval of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments

        TODO:
          Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
          Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
          inputs to None. This should be decided dynamically.
        """
        # Retrieve stored tensors
        epsilon = ctx.constant
        gamma, beta, sigma, x_hat = ctx.saved_tensors
        batch_size = x_hat.shape[0]

        # Gradient w.r.t. gamma
        grad_gamma = (x_hat * grad_output).sum(dim=0)

        # Gradient w.r.t. beta
        grad_beta = grad_output.sum(dim=0)

        # Gradient w.r.t. input
        grad_input = gamma * (sigma + epsilon)**(-1/2) / batch_size
        grad_input *= batch_size * grad_output - x_hat * grad_gamma - grad_beta

        # Return gradients, set gradient to None if ctx.needs_input_grad is False
        return_grads = [
            grad if ctx.needs_input_grad[i] else None
            for i, grad in enumerate([grad_input, grad_gamma, grad_beta])
        ]

        # return gradients of the three tensor inputs and None for the constant eps
        return tuple(return_grads + [None])


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormManualModule, self).__init__()

        self.n_neurons = n_neurons
        self.epsilon = eps

        self.gamma = nn.Parameter(torch.ones(1, n_neurons))
        self.beta = nn.Parameter(torch.zeros(1, n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        assert input.shape[1] == self.n_neurons

        custom_batch_norm_func = CustomBatchNormManualFunction()
        normalized = custom_batch_norm_func.apply(input, self.gamma, self.beta, self.epsilon)

        return normalized
