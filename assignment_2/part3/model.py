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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.device = device
        self.embedding_dim = 100
        self.lstm_num_hidden = lstm_num_hidden

        # Define architecture
        self.embeddings = nn.Embedding(vocabulary_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, lstm_num_hidden, lstm_num_layers)
        self.projection_layer = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, indices, cell=None):
        # Get embeddings
        x = self.embeddings(indices)
        x = x.unsqueeze(0)  # Add sequence "length" 1

        out, cell = self.lstm(x, cell)
        out = out.squeeze(0)
        p = self.projection_layer(out)

        return p, cell
