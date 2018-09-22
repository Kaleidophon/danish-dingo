"""
Create certain plots requires for assignment 2.
"""

# EXT
import matplotlib.pyplot as plt


def plot_length_accuracy(accuracies, save_dont_show=None):
    plt.plot(accuracies, color="red")

    plt.xlabel("Sequence length")
    plt.xticks(range(0, len(accuracies), 5), range(2, len(accuracies) + 2, 5))
    plt.ylabel("Accuracy")

    if save_dont_show is None:
        plt.show()
    else:
        plt.savefig(save_dont_show)
        plt.close()


if __name__ == "__main__":
    accuracies_by_length_vanilla = [
        0.9765625, 0.984375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2109375, 1.0, 1.0,
        1.0, 0.2109375, 0.2421875, 0.203125, 0.203125, 1.0, 0.21875, 0.2265625, 0.2109375, 0.21875, 0.203125,
        0.2109375, 0.3125, 0.2109375, 0.2265625, 0.2109375, 0.203125, 0.203125, 0.2265625, 0.2109375, 0.2109375,
        0.2109375, 0.2265625, 0.203125, 0.21875, 0.203125, 0.2109375, 0.21875, 0.203125, 0.21875, 0.2109375
    ]
    accuracies_by_length_lstm = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.796875, 0.21875, 1.0, 1.0, 1.0, 1.0, 0.21875, 0.2109375, 1.0, 1.0,
        0.2109375, 1.0, 0.2109375, 0.609375, 0.2109375, 1.0, 0.21875, 0.2109375, 0.203125, 0.203125, 1.0
    ]
    plot_length_accuracy(accuracies_by_length_lstm)
