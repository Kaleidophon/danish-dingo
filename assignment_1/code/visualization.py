"""
Visualizing and plotting results.
"""

# EXT
from scipy.interpolate import spline
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(batch_losses, epoch_losses):
    """
    Plot the training losses per batch / epoch jointly in one plot.
    """
    batch_size = int(len(batch_losses) / len(epoch_losses))
    epoch_losses = [batch_losses[0]] + epoch_losses  # Add first ever loss for visual ease

    # Interpolate epoch losses within epochs
    smoothed_axis = np.array(range(len(batch_losses)))
    points_with_data = np.array(range(len(epoch_losses))) * batch_size
    smoothed_epoch_losses = spline(points_with_data, epoch_losses, smoothed_axis, order=5)

    # Plot
    plt.plot(batch_losses, color="lightblue", label="Batch Loss")
    plt.plot(smoothed_epoch_losses, color="red", label="Average Loss")

    plt.xticks(range(0, len(batch_losses), batch_size), range(1, len(epoch_losses) + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc="upper right")

    plt.show()


def plot_accuracy(accuracies, eval_interval):
    """
    Plot the test set accuracies per epoch during training.
    """
    locs = np.array(range(0, len(accuracies))) * eval_interval

    plt.plot(locs, accuracies, color="orange")
    plt.xlabel("# Batch")
    plt.ylabel("Test Set Accuracy")

    plt.show()
