import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.bmnist import bmnist
import numpy as np


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear = nn.Linear(28*28, hidden_dim)
        self.mean_lin = nn.Linear(hidden_dim, z_dim)
        self.std_lin = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        encoded = self.linear(input)
        encoded = F.relu(encoded)

        mean = self.mean_lin(encoded)
        log_std = self.std_lin(encoded)
        std = torch.exp(log_std)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 28*28)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        h = self.linear1(input)
        h = F.relu(h)
        decoded = self.linear2(h)
        decoded = F.sigmoid(decoded)

        return decoded


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder.forward(input)
        epsilon = torch.FloatTensor(np.random.normal(0, 1, mean.size()))
        z = mean + epsilon * std
        decoded = self.decoder.forward(z)

        # Calculate loss
        bernoulli_loss = input * torch.log(decoded) + (1 - input) * torch.log(1 - decoded)
        bernoulli_loss = -torch.sum(bernoulli_loss, dim=1)
        bernoulli_loss = torch.sum(bernoulli_loss, dim=0)
        kl_loss = -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2))
        average_negative_elbo = bernoulli_loss + kl_loss

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None
        raise NotImplementedError()

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0

    for input in data:
        batch_size = input.size()[0]
        input = input.view(batch_size, 28 * 28)
        elbo = model.forward(input)
        elbo /= batch_size

        if model.training:
            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

        average_epoch_elbo += elbo

    average_epoch_elbo /= len(data)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch:<2}] train elbo: {train_elbo:.4f} val_elbo: {val_elbo:.4f}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
