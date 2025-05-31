import torch
import torch.nn as nn

import pickle
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch
import numpy as np



class CEMModel(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_layers=1):
        super(CEMModel, self).__init__()

        # First LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Second LSTM
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer for confidence score prediction
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional LSTM

        # Sigmoid activation for confidence estimation (output between 0 and 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        """
        x: Padded input tensor (batch_size, max_seq_len, input_dim)
        lengths: Actual lengths of sequences before padding
        """
        # Pack padded sequence
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # First LSTM
        x, _ = self.lstm1(x)

        # Second LSTM (directly taking output of first LSTM)
        x, _ = self.lstm2(x)

        # Unpack sequence
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Fully connected layer
        x = self.fc(x).squeeze(-1)  # Shape: (batch_size, max_seq_len)

        # Sigmoid activation
        return self.sigmoid(x)

