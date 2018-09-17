"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
from torch import nn, optim
from torch.autograd import Variable
from operator import mul
from functools import reduce

from visualization import plot_losses, plot_accuracy, write_data_to_file, plot_test_loss

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

# Determine if GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtypel = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """
    accurate_predictions = predictions.argmax(dim=1) == targets
    acc = float(accurate_predictions.cpu().float().mean(dim=0).numpy())
    return acc


def train():
    """
    Performs training and evaluation of ConvNet model.
    """
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Prepare data
    cifar10 = cifar10_utils.get_cifar10()
    train_set = cifar10["train"]
    test_set = cifar10["test"]
    x_test, y_test = test_set.images, test_set.labels
    n_channels = x_test.shape[1]
    n_classes = y_test.shape[1]
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], reduce(mul, x_test.shape[2:]))
    x_test = torch.Tensor(x_test).type(dtype)
    y_test = torch.Tensor(y_test).argmax(dim=1).type(dtypel)

    # Initialize model
    conv_net = ConvNet(n_channels, n_classes).to(device)

    # Prepare for training
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizer = optim.Adam(conv_net.parameters(), lr=LEARNING_RATE_DEFAULT)
    batch_nr_total = 0
    acc = 0
    batch_nr = 0
    num_batches = int(np.ceil(train_set.images.shape[0]) / BATCH_SIZE_DEFAULT)  # Number of batches per epoch
    epochs = int(np.ceil(MAX_STEPS_DEFAULT / num_batches))

    # Initialize data collection
    batch_losses = []
    current_epoch_losses = []
    test_losses = []
    epoch_losses = []
    all_accuracies = []

    while batch_nr_total < MAX_STEPS_DEFAULT:
        completed_epochs = train_set.epochs_completed
        x, y = train_set.next_batch(BATCH_SIZE_DEFAULT)
        #x = x.reshape(x.shape[0], x.shape[1], reduce(mul, x.shape[2:]))
        x = Variable(torch.Tensor(x).type(dtype))
        y = Variable(torch.Tensor(y).argmax(dim=1).type(dtypel))
        # Forward pass and loss
        optimizer.zero_grad()
        out = conv_net.forward(x).type(dtype)
        loss = loss_func(out, y)
        loss_ = float(loss.cpu().detach().numpy())
        batch_losses.append(loss_)
        current_epoch_losses.append(loss_)

        # Backward pass, adjust parameters
        loss.backward()
        optimizer.step()

        # Compute accuracy, print loss
        if batch_nr_total % EVAL_FREQ_DEFAULT == 0:
            acc, test_loss = eval_model(conv_net, loss_func, x_test, y_test)
            test_losses.append(test_loss)
            all_accuracies.append(acc)

        print("\r[Epoch {:>2}/{:>2} | Batch #{:>4}/{}] Loss: {:.2f} | Test Accuracy: {:.4f}".format(
            completed_epochs + 1, epochs, batch_nr, num_batches, loss_, acc
        ), end="", flush=True
        )

        # Prepare for next iteration
        batch_nr_total += 1

        # Reset batch counter if a new epoch has started
        if train_set.epochs_completed > completed_epochs:
            print("")
            batch_nr = 1
            epoch_losses.append(sum(current_epoch_losses) / len(current_epoch_losses))
            current_epoch_losses = []
        else:
            batch_nr += 1

    acc, test_loss = eval_model(conv_net, loss_func, x_test, y_test)
    test_losses.append(test_loss)
    all_accuracies.append(acc)
    print("\nTraining finished, final test accuracy is {:.4f}\n".format(acc))

    # Write to file just in case
    write_data_to_file(batch_losses, "./batch_losses.txt")
    write_data_to_file(epoch_losses, "./epoch_losses.txt")
    write_data_to_file(test_losses, "./test_losses.txt")
    write_data_to_file(all_accuracies, "./accuracies.txt")

    # Plot
    plot_losses(batch_losses, epoch_losses, save_dont_show="./train_losses.png")
    plot_accuracy(all_accuracies, EVAL_FREQ_DEFAULT, save_dont_show="./accuracies.png")
    plot_test_loss(test_losses, EVAL_FREQ_DEFAULT, save_dont_show="./test_losses.png")


def eval_model(model, loss_func, x_test, y_test, batch_size=BATCH_SIZE_DEFAULT):
    with torch.no_grad():
        test_predictions = []

        # Split to avoid memory errors -> Tensor with batch dimensionality of 10k might be too much
        for test_batch in torch.split(x_test, batch_size, dim=0):
            batch_predictions = model.forward(test_batch).type(dtype)
            test_predictions.append(batch_predictions)

        test_predictions = torch.cat(test_predictions, dim=0)
        acc = accuracy(test_predictions, y_test)
        test_loss = loss_func(test_predictions, y_test)
        test_loss = float(test_loss.cpu().detach().numpy())

        return acc, test_loss


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    print("Device: {}".format(device))

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
