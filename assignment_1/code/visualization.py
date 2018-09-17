"""
Visualizing and plotting results.
"""

# EXT
import matplotlib
matplotlib.use('Agg')  # Make it compatible with cluster

from scipy.interpolate import spline
import numpy as np
import matplotlib.pyplot as plt
import codecs


def plot_losses(batch_losses, epoch_losses, save_dont_show=None):
    """
    Plot the training losses per batch / epoch jointly in one plot.
    """
    batch_size = int(len(batch_losses) / len(epoch_losses))
    epoch_losses = [batch_losses[0]] + epoch_losses  # Add first ever loss for visual ease

    # Interpolate epoch losses within epochs
    smoothed_axis = np.array(range(len(batch_losses)))
    points_with_data = np.array(range(len(epoch_losses))) * batch_size
    smoothed_epoch_losses = spline(points_with_data, epoch_losses, smoothed_axis, order=2)

    # Plot
    plt.plot(batch_losses, color="lightblue", label="Batch Loss")
    plt.plot(smoothed_epoch_losses, color="red", label="Average Loss")

    plt.xticks(range(0, len(batch_losses), batch_size), range(1, len(epoch_losses) + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc="upper right")

    if save_dont_show is None:
        plt.show()
    else:
        plt.savefig(save_dont_show)
        plt.close()


def plot_test_loss(losses, eval_interval, save_dont_show=None):
    locs = np.array(range(0, len(losses))) * eval_interval
    plt.plot(locs, losses, color="red")

    plt.xlabel("# Batch")
    plt.ylabel("Test Set Loss")

    if save_dont_show is None:
        plt.show()
    else:
        plt.savefig(save_dont_show)
        plt.close()


def plot_accuracy(accuracies, eval_interval, save_dont_show=None, y_limits=None):
    """
    Plot the test set accuracies per epoch during training.
    """
    locs = np.array(range(0, len(accuracies))) * eval_interval

    if y_limits is not None:
        plt.ylim(*y_limits)

    plt.plot(locs, accuracies, color="orange")
    plt.xlabel("# Batch")
    plt.ylabel("Test Set Accuracy")

    if save_dont_show is None:
        plt.show()
    else:
        plt.savefig(save_dont_show)
        plt.close()


def write_data_to_file(data, path):
    with codecs.open(path, "wb", "utf-8") as file:
        for entry in data:
            file.write("{}\n".format(entry))


def read_data_from_file(path):
    with codecs.open(path, "rb", "utf-8") as file:
        return list(map(lambda l: float(l.strip()), file.readlines()))
