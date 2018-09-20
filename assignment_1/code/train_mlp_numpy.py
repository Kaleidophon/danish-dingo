"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

from operator import mul
from functools import reduce

from visualization import plot_losses, plot_accuracy, plot_test_loss

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

# Custom constants
N_INPUTS = 3072
N_CLASSES = 10
EPOCHS_DEFAULT = 10

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
    acc = (predictions.argmax(axis=1) == targets.argmax(axis=1))
    acc = acc.astype(int)
    acc = acc.mean()

    return acc


def train(max_steps=MAX_STEPS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT, learning_rate=LEARNING_RATE_DEFAULT,
          eval_interval=EVAL_FREQ_DEFAULT):
    """
    Performs training and evaluation of MLP model.
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

    # Initialize model
    nn = MLP(n_inputs=N_INPUTS, n_hidden=dnn_hidden_units, n_classes=N_CLASSES)
    loss_func = CrossEntropyModule()
    batch_nr = 1
    batch_nr_total = 0
    acc = 0
    test_loss = -1

    # Initialize data collection
    batch_losses = []
    current_epoch_losses = []
    epoch_losses = []
    all_accuracies = []
    test_losses = []

    num_batches = int(np.ceil(train_set.images.shape[0]) / batch_size)  # Number of batches per epoch
    epochs = int(np.ceil(max_steps / num_batches))

    while batch_nr_total < MAX_STEPS_DEFAULT:

        completed_epochs = train_set.epochs_completed
        x, y = train_set.next_batch(BATCH_SIZE_DEFAULT)
        x = x.reshape(batch_size, reduce(mul, x_test.shape[1:]))

        # Forward pass and loss
        out = nn.forward(x)
        loss = loss_func.forward(out, y)
        batch_losses.append(loss)
        current_epoch_losses.append(loss)

        # Backward pass
        loss_gradient = loss_func.backward(out, y)
        nn.backward(loss_gradient)

        # Adjust parameters
        for layer in nn.layers:
            if isinstance(layer, LinearModule):
                layer.update_parameters(learning_rate)

        # Compute accuracy, print loss
        if batch_nr_total % EVAL_FREQ_DEFAULT == 0:
            acc, test_loss = eval_model(nn, loss_func.forward, x_test, y_test)
            all_accuracies.append(acc)
            test_losses.append(test_loss)

        print(
            "\r[Epoch {:>2}/{:>2} | Batch #{:>3}] Train Loss: {:.2f}, Test Loss {:.2f} | Test Accuracy: {:.4f}".format(
                completed_epochs + 1, epochs, batch_nr, loss, test_loss, acc
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

    # Last evaluation
    acc, test_loss = eval_model(nn, loss_func.forward, x_test, y_test)
    test_losses.append(test_loss)
    all_accuracies.append(acc)
    print("\nTraining finished, final test accuracy is {:.4f}\n".format(acc))

    # Plotting
    plot_losses(batch_losses=batch_losses, epoch_losses=epoch_losses, save_dont_show="./train_losses_mlp_np.png")
    plot_accuracy(all_accuracies, eval_interval, save_dont_show="./accuracy_mlp_np.png", y_limits=(0.10, 0.55))
    plot_test_loss(test_losses, save_dont_show="./test_losses_mlp_np.png", eval_interval=EVAL_FREQ_DEFAULT)


def eval_model(model, loss_func, x_test, y_test):
        test_predictions = model.forward(x_test)
        acc = accuracy(test_predictions, y_test)
        test_loss = loss_func(test_predictions, y_test)
        test_loss = float(test_loss)

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
