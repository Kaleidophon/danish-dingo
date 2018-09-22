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


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.device = device

        # Define parameters
        # LSTM cell
        self.W_gx = nn.Parameter(self._init_weights(input_dim, num_hidden, device=device))
        self.W_gh = nn.Parameter(self._init_weights(num_hidden, num_hidden, device=device))
        self.b_g = nn.Parameter(torch.zeros(num_hidden, device=device))
        self.W_ix = nn.Parameter(self._init_weights(input_dim, num_hidden, device=device))
        self.W_ih = nn.Parameter(self._init_weights(num_hidden, num_hidden, device=device))
        self.b_i = nn.Parameter(torch.zeros(num_hidden, device=device))
        self.W_fx = nn.Parameter(self._init_weights(input_dim, num_hidden, device=device))
        self.W_fh = nn.Parameter(self._init_weights(num_hidden, num_hidden, device=device))
        self.b_f = nn.Parameter(torch.zeros(num_hidden, device=device))
        self.W_ox = nn.Parameter(self._init_weights(input_dim, num_hidden, device=device))
        self.W_oh = nn.Parameter(self._init_weights(num_hidden, num_hidden, device=device))
        self.b_o = nn.Parameter(torch.zeros(num_hidden, device=device))

        # Normal RNN part
        self.W_ph = nn.Parameter(self._init_weights(num_hidden, num_classes, device=device))
        self.b_p = nn.Parameter(torch.zeros(num_classes, device=device))

        # Define rest
        self._reset_states()
        self.t = 0

    def forward(self, x):
        g_t = F.tanh(x @ self.W_gx + self.h_previous @ self.W_gh + self.b_g)
        i_t = F.sigmoid(x @ self.W_ix + self.h_previous @ self.W_ih + self.b_i)
        f_t = F.sigmoid(x @ self.W_fx + self.h_previous @ self.W_fh + self.b_f)
        o_t = F.sigmoid(x @ self.W_ox + self.h_previous @ self.W_oh + self.b_o)
        c_t = g_t * i_t + self.c_previous * f_t
        h_t = F.tanh(c_t) * o_t

        p_t = h_t @ self.W_ph + self.b_p

        # Prepare for next iteration
        self.t += 1

        # Store hidden state for next input symbol
        if self.t < self.seq_length:
            self.h_previous = h_t
            self.c_previous = c_t

        # Reset to initial hidden state if end of sequence was reached
        else:
            self._reset_states()
            self.t = 0

        return p_t

    @staticmethod
    def _init_weights(*shape, device):
        return torch.randn(*shape).to(device) / 5  # mu = 0, sigma^2 = 0.2

    def _reset_states(self):
        self.h_previous = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        self.c_previous = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
