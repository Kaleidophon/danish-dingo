################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.input_dim = input_dim

        # Define parameters
        self.W_hx = nn.Parameter(torch.rand(input_dim, num_hidden)).to(device)
        self.W_hh = nn.Parameter(torch.rand(num_hidden, num_hidden)).to(device)
        self.b_h = nn.Parameter(torch.zeros(num_hidden)).to(device)
        self.W_ph = nn.Parameter(torch.rand(num_hidden, self.num_classes)).to(device)
        self.b_p = nn.Parameter(torch.zeros(num_classes)).to(device)

        # Define rest
        self.h_previous = torch.zeros(batch_size, num_hidden)
        self.t = 0

    def forward(self, x):
        h_t = F.tanh(x @ self.W_hx + self.h_previous @ self.W_hh + self.b_h)
        p_t = h_t @ self.W_ph + self.b_p

        # Prepare for next iteration
        self.t += 1

        # Store hidden state for next input symbol
        if self.t < self.seq_length:
            self.h_previous = h_t
        # Reset to initial hidden state if end of sequence was reached
        else:
            self.h_previous = torch.zeros(self.batch_size, self.num_hidden)
            self.t = 0

        return p_t
