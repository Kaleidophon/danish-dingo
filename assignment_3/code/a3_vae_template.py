import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.bmnist import bmnist
import numpy as np
from scipy.stats import norm


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
        kl_loss = -0.5 * torch.sum(1 + torch.log(std.pow(2) + 1e-5) - mean.pow(2) - std.pow(2))
        average_negative_elbo = bernoulli_loss + kl_loss

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        with torch.no_grad():
            sampled_z = torch.FloatTensor(np.random.normal(0, 1, (n_samples, self.z_dim)))
            sampled_ims = self.decoder.forward(sampled_z)
            sampled_ims = sampled_ims.view(n_samples, 1, 28, 28)

        im_means = sampled_ims.mean(dim=1)
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


def plot_sampled_images(images, filename):
    grid = make_grid(images, padding=1, nrow=images.size()[0])
    npimg = grid.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)


def plot_manifold(model, filename):
    n_rows = 20

    x_coords = np.linspace(-1, 1, n_rows)
    unit_square = np.transpose([np.tile(x_coords, len(x_coords)), np.repeat(x_coords, len(x_coords))])
    x_coords_trans = np.linspace(norm.ppf(0.01), norm.ppf(0.99), n_rows)
    unit_square_trans = np.transpose(
        [np.tile(x_coords_trans, len(x_coords_trans)), np.repeat(x_coords_trans, len(x_coords_trans))])

    # Visualize learned data manifold with transformed unit square coordinates
    fig = plt.figure(figsize=(10, 10))

    for i, coords in enumerate(unit_square_trans):
        plt.subplot(n_rows, n_rows, i + 1)
        plt.axis("off")

        coords = torch.FloatTensor(coords).unsqueeze(0)
        decoded = model.decoder(coords)
        decoded *= -1  # Invert colors
        plt.imshow(decoded.view(28, 28).data.numpy(), cmap='gist_gray')

    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    n_samples = 5
    sample_epochs = (0, 4, 9, 14, 19, 24, 29, 34, 39)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch:<2}] train elbo: {train_elbo:.4f} val_elbo: {val_elbo:.4f}")
        sampled_images, _ = model.sample(n_samples=n_samples)

        if epoch in sample_epochs:
            plot_sampled_images(sampled_images, "./sampled_digits/sampled_{}_epoch_{}.png".format(n_samples, epoch + 1))

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')

    if ARGS.zdim == 2:
        plot_manifold(model, "./manifold.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
