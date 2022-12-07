import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from timeit import default_timer
import operator
from functools import reduce
from functools import partial
from matplotlib import cm
import scipy.io
from utilities3 import LpLoss
from normal import UnitGaussianNormalizer
import sys
sys.path.append("../gru")
sys.path.append("../fno")
sys.path.append("../lstm")
sys.path.append("../deeponet")
from gru2 import GRUModel
from lstm2 import LSTMModel
from fourier import FNO1d
from deeponet import DeepONet
from utilities3 import LpLoss

linestyle_tuple = [
             ('loosely dotted',        (0, (1, 10))),
                  ('dotted',                (0, (1, 1))),
                       ('densely dotted',        (0, (1, 1))),
                            ('long dash with offset', (5, (10, 3))),
                                 ('loosely dashed',        (0, (5, 10))),
                                      ('dashed',                (0, (5, 5))),
                                           ('densely dashed',        (0, (5, 1))),

                                                ('loosely dashdotted',    (0, (3, 10, 1, 10))),
                                                     ('dashdotted',            (0, (3, 5, 1, 5))),
                                                          ('densely dashdotted',    (0, (3, 1, 1, 1))),

                                                               ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                                                                    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                                                                         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]




# Model parameters
ntrain = 900
ntest = 100
batch_size = 20
gamma = 0.5 
learning_rate= 0.001
step_size = 50
modes = 12
width = 32
hidden_units = 400

# Data Parameters
T = 1
dt = 0.01
nt = int(round(T/dt))
npoints = 1000
t1 = np.linspace(0.00, T, nt, dtype=np.float32)

# Build Dataset
x = np.loadtxt("x.dat", dtype=np.float32)
x = x.reshape(
            x.shape[0], x.shape[1] // 2, 2)
y = np.loadtxt("y.dat", dtype=np.float32)
y = y.reshape(
            y.shape[0], y.shape[1] // 1, 1)

# Build grid for deeponet
grid_repeated = []
for i in range(batch_size):
    grid_repeated.append(t1.reshape(100, 1))
grid_repeated = torch.from_numpy(np.array(grid_repeated)).cuda()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# Build models
fnoModel = FNO1d(modes, width, 2, 1).cuda()
gruModel = GRUModel(hidden_units, 2, 1).cuda()
lstmModel = LSTMModel(hidden_units, 2, 1).cuda()

branch = nn.Sequential(GRUModel(100, 2, 256))
trunk = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
deepModel = DeepONet(branch, trunk).cuda()

fnoModel.load_state_dict(torch.load('fnoModel1'))
gruModel.load_state_dict(torch.load('gruModel1'))
lstmModel.load_state_dict(torch.load('lstmModel1'))
deepModel.load_state_dict(torch.load('deeponet1'))

gruModel.eval()
fnoModel.eval()
lstmModel.eval()
deepModel.eval()

idx=  0
idx2 = 0
with torch.no_grad():
    for x, y in testData:
        x, y = x.cuda(), y.cuda()   
        outGRU = gruModel(x)
        outLSTM = lstmModel(x)
        outFNO = fnoModel(x)
        outFNO = outFNO.reshape((outFNO.shape[0], outFNO.shape[1]))
        outDeep = deepModel(x, grid_repeated)
        fig, axes = plt.subplots()
        axes.plot(t1, y[0].detach().cpu(), label="Original Parameter Estimator")
        axes.plot(t1, outGRU[0].detach().cpu(), label="GRU", linestyle=linestyle_tuple[1][1])
        axes.plot(t1, outLSTM[0].detach().cpu(), label="LSTM", linestyle=linestyle_tuple[1][1])
        axes.plot(t1, outDeep[0].detach().cpu(), label="DeepONet", linestyle=linestyle_tuple[1][1])
        axes.plot(t1, outFNO[0].detach().cpu(), label="FNO", linestyle=linestyle_tuple[1][1])
        inset = axes.inset_axes([0.7, 0.4, 0.2, 0.2])
        inset.plot(t1, y[0].detach().cpu())
        inset.plot(t1, outGRU[0].detach().cpu(), linestyle=linestyle_tuple[1][1])
        inset.plot(t1, outLSTM[0].detach().cpu(), linestyle=linestyle_tuple[1][1])
        inset.plot(t1, outDeep[0].detach().cpu(), linestyle=linestyle_tuple[1][1])
        inset.plot(t1, outFNO[0].detach().cpu(), linestyle=linestyle_tuple[1][1])
        inset.set_xlim(0.6, 0.8)
        if idx == 1:
            inset.set_ylim(y[0][59].detach().cpu().numpy().item()-0.01, y[0][59].detach().cpu().numpy().item()+0.01)
        elif idx == 2:
            inset.set_ylim(y[0][59].detach().cpu().numpy().item()-0.01, y[0][59].detach().cpu().numpy().item()+0.01)
        elif idx == 3:
            inset.set_ylim(y[0][59].detach().cpu().numpy().item()-0.01, y[0][59].detach().cpu().numpy().item()+0.01)
        elif idx==4:
            inset.set_ylim(y[0][59].detach().cpu().numpy().item()-0.1, y[0][59].detach().cpu().numpy().item()+0.1)
        inset.set_xticklabels([])
        inset.set_yticklabels([])
        inset.set_xticks([], minor=True)
        inset.set_yticks([], minor=True)
        inset.tick_params(left=False, bottom=False)
        rect, line = axes.indicate_inset_zoom(inset, alpha=1)
        rect.set_edgecolor('none')
        axes.set_xlabel("t")
        axes.set_ylabel(r"Estimate of $\hat{\theta}$")
        plt.legend()
        plt.savefig(str(idx) + ".eps")
        idx+=1

with torch.no_grad():
    testLossfno = 0
    testLosslstm = 0
    testLossgru = 0
    testLossdeep = 0
    loss = LpLoss()
    for x, y in testData:
        x, y = x.cuda(), y.cuda()
        outGRU = gruModel(x)
        outLSTM = lstmModel(x)
        outFNO = fnoModel(x)
        outFNO = outFNO.reshape((outFNO.shape[0], outFNO.shape[1]))
        outDeep = deepModel(x, grid_repeated)

        testLossfno += loss(outFNO, y).item()
        testLosslstm += loss(outLSTM, y).item()
        testLossgru += loss(outGRU, y).item()
        testLossdeep += loss(outDeep, y).item()

print("Total Relative Error Testing Set FNO (100 Points)", testLossfno)
print("Total Relative Error Testing Set GRU (100 Points)", testLossgru)
print("Total Relative Error Testing Set LSTM (100 Points)", testLosslstm)
print("Total Relative Error Testing Set DeepONet (100 Points)", testLossdeep)
print("Average Relative Error Testing Set FNO (100 Points)", testLossfno/100)
print("Average Relative Error Testing Set GRU (100 Points)", testLossgru/100)
print("Average Relative Error Testing Set LSTM (100 Points)", testLosslstm/100)
print("Average Relative Error Testing Set DeepONet (100 Points)", testLossdeep/100)

with torch.no_grad():
    testLossfno = 0
    testLosslstm = 0
    testLossgru = 0
    testLossdeep = 0
    loss = LpLoss()
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()
        outGRU = gruModel(x)
        outLSTM = lstmModel(x)
        outFNO = fnoModel(x)
        outDeep = deepModel(x, grid_repeated)

        testLossfno += loss(outFNO, y).item()
        testLosslstm += loss(outLSTM, y).item()
        testLossgru += loss(outGRU, y).item()
        testLossdeep += loss(outDeep, y).item()

print("Total Relative Error Training Set FNO (900 Points)", testLossfno)
print("Total Relative Error Training Set GRU (900 Points)", testLossgru)
print("Total Relative Error Training Set LSTM (900 Points)", testLosslstm)
print("Total Relative Error Training Set DeepONet (900 Points)", testLossdeep)
print("Average Relative Error Training Set FNO (900 Points)", testLossfno/900)
print("Average Relative Error Training Set GRU (900 Points)", testLossgru/900)
print("Average Relative Error Training Set LSTM (900 Points)", testLosslstm/900)
print("Average Relative Error Training Set DeepONet (900 Points)", testLossdeep/900)

