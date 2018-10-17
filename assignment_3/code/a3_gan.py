import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
import random

# Determine if GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Generator(nn.Module):
    def __init__(self, latent_dim, negative_slope=0.2):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(negative_slope),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(negative_slope),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    criterion = nn.BCELoss()
    num_batches = len(dataloader)

    for epoch in range(args.n_epochs):
        for i, (images, _) in enumerate(dataloader):

            batch_size = images.size()[0]
            images = Variable(images.view(batch_size, 28 * 28)).to(device)
            latent_dim = generator.latent_dim

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            z = Variable(torch.randn(batch_size, latent_dim).to(device))  # Sample noise
            fake_images = generator(z)
            # Use noisy labels
            real_labels = torch.empty(batch_size, 1).uniform_(0.8, 1.1).to(device)
            fake_predictions = discriminator(fake_images)
            # How well did the generator fool the discriminator?
            generator_loss = criterion(fake_predictions, real_labels)
            generator_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            # Use noisy labels
            fake_labels = torch.empty(batch_size, 1).uniform_(0, 0.2).to(device)
            real_predictions = discriminator(images)
            # How well was the discriminator able to identify real images?
            real_loss = criterion(real_predictions, real_labels)
            # How well was the discriminator able to identify forged images?
            fake_predictions2 = discriminator(fake_images.detach())
            fake_loss = criterion(fake_predictions2, fake_labels)
            discriminator_loss = 0.5 * (real_loss + fake_loss)
            discriminator_loss.backward()
            optimizer_D.step()

            # Screen output
            if (i + 1) % args.print_interval == 0:
                print(
                    "\r[Epoch {:>2}/{:>2}, Batch {:>3}/{:>3}] || Generator loss: {:.4f} | Discriminator loss: {:.4f}".format(
                        epoch+1, args.n_epochs, i+1, num_batches, generator_loss, discriminator_loss
                    ), end="\n" if (i + 1) % (args.print_interval * 10) == 0 and i != 0 else "", flush=True
                )

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                fake_images = fake_images.view(batch_size, 1, 28, 28)
                save_image(
                    fake_images[:25], 'images/{}.png'.format(batches_done), nrow=5, normalize=True
                )

                # Interpolations
                # Do it during training due to some mysterious problems
                for inter in range(10):
                    # Sample two of the digits that just were generated
                    z1 = z[random.randint(0, batch_size-1), :]
                    z2 = z[random.randint(0, batch_size-1), :]

                    create_interpolations(
                        z1, z2, generator, discriminator,
                        figure_path="interpolation{}_epoch_{}_".format(inter, epoch) + "{}.png",
                    )


def interpolate_z(z1, z2, num_interpolations=7, latent_dim=100):
    diff = z2 - z1
    interpolations = [z1]

    for i in range(num_interpolations):
        interpolations.append(z1 + i / num_interpolations * diff)

    interpolations.append(z2)
    interpolations = torch.cat(interpolations, dim=0).view(num_interpolations + 2, latent_dim).to(device)

    return interpolations


def create_interpolations(z1, z2, generator, discriminator, figure_path, latent_dim=100, num_interpolations=7):
    generator.eval()
    discriminator.eval()

    with torch.no_grad():

        interpolated_zs = interpolate_z(z1, z2, num_interpolations=num_interpolations, latent_dim=latent_dim)
        generated_imgs = generator(interpolated_zs)
        generated_imgs = generated_imgs.view(num_interpolations + 2, 1, 28, 28)

        save_image(
            generated_imgs, figure_path, nrow=num_interpolations + 2, normalize=True
        )

    generator.train()
    discriminator.train()


def main(args):
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator, "mnist_generator.pt")
    torch.save(discriminator, "mnist_discriminator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--print_interval', type=int, default=100,
                        help='Print every PRINT_INTERVAL iterations')
    args = parser.parse_args()

    main(args)
