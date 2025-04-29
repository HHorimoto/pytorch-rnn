import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_hidden, rnn_name):
        super(Net, self).__init__()

        self.rnn_name = rnn_name

        if self.rnn_name == "RNN":
            self.rnn = nn.RNNCell(34, n_hidden, batch_first=True)
        elif rnn_name == "LSTM":
          self.rnn = nn.LSTMCell(34, n_hidden, batch_first=True)
        elif rnn_name == "GRU":
          self.rnn = nn.GRUCell(34, n_hidden, batch_first=True)
        
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x, hx, cx):
        if self.rnn_name == "RNN":
            hx = self.rnn(x, hx)
            h = self.linear(hx)
            return h, hx, cx
        elif self.rnn_name == "LSTM":
            hx, cx = self.rnn(x, (hx, cx))
            h = self.linear(hx)
            return h, hx, cx
        elif self.rnn_name == "GRU":
            hx = self.rnn(x, hx)
            h = self.linear(hx)
            return h, hx, cx