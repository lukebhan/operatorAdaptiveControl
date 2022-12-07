import torch.nn as nn
import torch.optim as optim
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layersLstm, num_layersLinear, sizeLayersLinear):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layersLstm = num_layersLstm
        self.num_layersLinear  = num_layersLinear
        self.sizeLayersLinear = sizeLayersLinear

        self.net = nn.Sequential()
        self.lstm = (nn.LSTM(self.input_size, self.hidden_size, self.num_layersLstm, batch_first=True))

        self.net.append(nn.Linear(self.hidden_size, self.sizeLayersLinear[0]))
        for i in range(self.num_layersLinear-1):
            self.net.append(nn.Linear(self.sizeLayersLinear[i], self.sizeLayersLinear[i+1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(self.sizeLayersLinear[-1], self.output_size))


    def forward(self, x):
        outputs = []
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layersLstm, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layersLstm, batch_size, self.hidden_size).cuda()
        output, (_, _) = self.lstm(x, (h0, c0))
        out = self.net(output)
        return out
