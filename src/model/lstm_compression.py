import torch
from torch import nn


class LSTM_fc(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_seq_len, output_size):
        super(LSTM_fc, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2 * hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        output = self.fc(out[:, :self.output_seq_len, :])

        return output
