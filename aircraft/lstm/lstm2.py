import torch.nn as nn
import torch.optim as optim
import torch

class LSTMModel(nn.Module):
    def __init__(self, width, x_size, y_size):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = width
        self.x_size = x_size

        self.lstm = nn.LSTM(input_size = x_size, hidden_size=width, num_layers=1, batch_first=True)
        self.fc = nn.Linear(width, y_size)
        
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out
