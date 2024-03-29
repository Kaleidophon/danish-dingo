# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
from random import choice

import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from part3.dataset import TextDataset
from part3.model import TextGenerationModel

################################################################################


def generate_sentence(model, seq_length, dataset, device, temperature=2, sampled_ch_idx=None, seed_phrase=None):
    """
    Generate a sequence with a LSTM model by sampling from character distributions.
    """
    with torch.no_grad():
        # Generate first character unless character has already been generated somewhere else
        # (The first character might be shared among sampling methods to ensure comparability)
        if sampled_ch_idx is None:
            index = choice(range(dataset.vocab_size))
        else:
            index = sampled_ch_idx

        initially_sampled = index
        generated = []
        cell = None

        # Use an initial string to "warm up" the model
        if seed_phrase is not None:
            for ch in seed_phrase:
                index = dataset._char_to_ix[ch]
                out, cell = model(torch.LongTensor([index]).to(device), cell)
            index = int(out.argmax().cpu().numpy())
        else:
            generated.append(dataset._ix_to_char[index])

        # Sample phrase
        for i in range(seq_length):
            out, cell = model(torch.LongTensor([index]).to(device), cell)

            # Greedy sampling
            if temperature is None:
                predicted = int(out.argmax().cpu().numpy())

            # Sampling with temperature
            else:
                out = F.softmax(out / temperature)
                dist = Categorical(out)
                predicted = int(dist.sample_n(1).cpu().numpy())

            generated.append(dataset._ix_to_char[predicted])
            index = predicted

    # Print results
    print_temp = "greedily" if temperature is None else "with temperature {}".format(temperature)
    sampled_phrase = "{}{}".format(
        seed_phrase if seed_phrase is not None else "",
        "".join(generated).replace("\r", "").replace("\n", " ")
    )
    print("Generated sentence {}: {}".format(print_temp, sampled_phrase))

    return initially_sampled


def calculate_accuracy(predictions, targets):
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


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length, )

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, device
    ).to(device)
    seq_length, batch_size = config.seq_length, config.batch_size

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    epoch = 0
    break_loop = False

    while True:
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1, shuffle=True, drop_last=True)
        num_batches = len(data_loader)

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()
            optimizer.zero_grad()
            loss = 0
            accuracy = 0
            cell = None  # Initial cell state

            for indices, targets in zip(batch_inputs, batch_targets):
                indices = Variable(indices).to(device)
                y = Variable(targets).to(device)
                out, cell = model(indices, cell)
                loss = criterion(out, y)
                accuracy += calculate_accuracy(out, y)

            # Reshape and calculate loss / accuracy
            loss /= seq_length
            loss.backward()
            accuracy /= seq_length
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if (step + epoch * num_batches) % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step + epoch * num_batches,
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if (step + epoch * num_batches) % config.sample_every == 0:
                # Generate sentence
                idx = None  # Generate sentences with the first initial character
                for temp in [None, 0.5, 1, 2]:
                    idx = generate_sentence(
                        model, seq_length, dataset, device, temperature=temp, sampled_ch_idx=idx, seed_phrase="North America "
                    )

            if (step + epoch * num_batches) == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break_loop = True
                break
        else:
            if break_loop:
                break

        epoch += 1

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temperature', type=float, default=None, help="Temperature for sampling when generating sentences.")

    config = parser.parse_args()
    print(config)

    # Train the model
    train(config)
