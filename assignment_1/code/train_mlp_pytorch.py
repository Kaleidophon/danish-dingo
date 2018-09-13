"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

from operator import mul
from functools import reduce
import torch
from torch import nn, optim

from visualization import plot_losses, plot_accuracy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


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
    accurate_predictions = predictions.argmax(dim=1) == targets.argmax(dim=1)
    accuracy = float(accurate_predictions.double().mean(dim=0).numpy())
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # Prepare data
    cifar10 = cifar10_utils.get_cifar10()
    train_set = cifar10["train"]
    test_set = cifar10["test"]
    x_test, y_test = test_set.images, test_set.labels
    x_test = x_test.reshape(x_test.shape[0], reduce(mul, x_test.shape[1:]))
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    # Initialize model
    n_inputs = x_test.shape[1]
    n_classes = y_test.shape[1]
    mlp = MLP(n_inputs=n_inputs, n_hidden=dnn_hidden_units, n_classes=n_classes)

    # Prepare for training
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE_DEFAULT)
    batch_nr_total = 0
    test_predictions = mlp.forward(x_test)
    acc = accuracy(test_predictions, y_test)
    batch_nr = 0
    num_batches = int(np.ceil(train_set.images.shape[0]) / BATCH_SIZE_DEFAULT)  # Number of batches per epoch
    epochs = int(np.floor(MAX_STEPS_DEFAULT / num_batches))

    # Initialize data collection
    batch_losses = []
    current_epoch_losses = []
    epoch_losses = []
    all_accuracies = []

    while batch_nr_total < MAX_STEPS_DEFAULT:

        completed_epochs = train_set.epochs_completed
        x, y = train_set.next_batch(BATCH_SIZE_DEFAULT)
        x = torch.Tensor(x.reshape(BATCH_SIZE_DEFAULT, reduce(mul, x_test.shape[1:])))
        y = torch.LongTensor(y).argmax(dim=1)

        # Forward pass and loss
        out = mlp.forward(x)
        loss = loss_func(out, y)
        loss_ = float(loss.detach().numpy())
        batch_losses.append(loss_)
        current_epoch_losses.append(loss_)

        # Backward pass, adjust parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute accuracy, print loss
        if batch_nr_total % EVAL_FREQ_DEFAULT == 0:
            with torch.no_grad():
                test_predictions = mlp.forward(x_test)
                acc = accuracy(test_predictions, y_test)
                all_accuracies.append(acc)

        print("\r[Epoch {:>2}/{:>2} | Batch #{:>3}] Loss: {:.2f} | Test Accuracy: {:.4f}".format(
                completed_epochs + 1, epochs, batch_nr, loss_, acc
            ), end="", flush=True
        )

        # Prepare for next iteration
        batch_nr_total += 1

        # Reset batch counter if a new epoch has started
        if train_set.epochs_completed > completed_epochs:
            print("")
            batch_nr = 0
            completed_epochs = train_set.epochs_completed
            epoch_losses.append(sum(current_epoch_losses) / len(current_epoch_losses))
            current_epoch_losses = []
        else:
            batch_nr += 1

    plot_losses(batch_losses=batch_losses, epoch_losses=epoch_losses)
    plot_accuracy(all_accuracies, EVAL_FREQ_DEFAULT)

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

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
