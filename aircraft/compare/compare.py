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
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import sys
sys.path.append("../gru")
sys.path.append("../fno")
sys.path.append("../lstm")
sys.path.append("../deeponet")
from gru2 import GRUModel
from lstm2 import LSTMModel
from deeponet import DeepONet2D
from fourier import FNO1d
from utilities3 import LpLoss


font = {'family': 'normal', 'weight':'bold', 'size':12}
matplotlib.rc('font', **font)



# Model parameters
ntrain = 900
ntest = 100
batch_size = 20
gamma = 0.5 
learning_rate= 0.001
step_size = 50
modes = 16
width = 48
hidden_units = 850

# Data Parameters
T = 20
dt = 0.01
nt = int(round(T/dt))
npoints = 1000

t1 = np.linspace(0.00, T, nt, dtype=np.float32)

grid_repeated = []
for i in range(batch_size):
    grid_repeated.append(t1.reshape((2000, 1)))
grid_repeated = torch.from_numpy(np.array(grid_repeated)).cuda()

x = np.loadtxt("xAlpha.dat", dtype=np.float32)
x = x.reshape(
            x.shape[0], x.shape[1] // 7, 7)
y = np.loadtxt("y.dat", dtype=np.float32)
y = y.reshape(
            y.shape[0], y.shape[1] // 5, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()
testDataNoNormal = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
trainDataNoNormal = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

x_normalizer = UnitGaussianNormalizer(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
x_train  = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_train = y_normalizer.encode(y_train)

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# Build models
fnoModel = FNO1d(modes, width, 7, 5).cuda()
gruModel = GRUModel(hidden_units, 7, 5).cuda()
lstmModel = LSTMModel(hidden_units, 7, 5).cuda()
branch = torch.nn.Sequential(GRUModel(200, 7, 500)).cuda()
trunk = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.ReLU(), torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 500), torch.nn.ReLU()).cuda()
deepModel = DeepONet2D(branch, trunk, 5).cuda()

fnoModel.load_state_dict(torch.load('fnoModel1'))
gruModel.load_state_dict(torch.load('gruModel1'))
lstmModel.load_state_dict(torch.load('lstmModel1'))
deepModel.load_state_dict(torch.load('deeponet1'))
gruModel.eval()
fnoModel.eval()
lstmModel.eval()
deepModel.eval()
idx=  0
with torch.no_grad():
    for x, y in testData:
        x, y = x.cuda(), y.cuda()   
        outGRU = gruModel(x)
        outLSTM = lstmModel(x)
        outFNO = fnoModel(x)

        outFNO = y_normalizer.decode(outFNO)
        outLSTM = y_normalizer.decode(outLSTM)
        outGRU = y_normalizer.decode(outGRU)

with torch.no_grad():
    for x, y in testDataNoNormal:
        x, y = x.cuda(), y.cuda()
        outDeep = deepModel(x, grid_repeated)

fig, ax = plt.subplots(5, 5, figsize=(20, 15))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
for j in range(5):
    ax[j][0].set_ylabel(r"$\hat{\theta}_" + str(j+1) + "$(t)", font)
    ax[0][j].set_title("Example " + str(j+1), font)
    for i in range(5):
        ax[i][j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        l1, = ax[i][j].plot(t1, y[j,:,i].detach().cpu(), label="Original Parameter Estimator")
        l2, = ax[i][j].plot(t1, outGRU[j,:,i].detach().cpu(), label="GRU")
        l3, = ax[i][j].plot(t1, outLSTM[j,:,i].detach().cpu(), label="LSTM")
        l4, = ax[i][j].plot(t1, outDeep[j,:,i].detach().cpu(), label="DeepONet")
        l5, = ax[i][j].plot(t1, outFNO[j,:,i].detach().cpu(), label="FNO")
    ax[-1][j].set_xlabel("t", font)

plt.legend([l1, l2,l3,  l4, l5], ["Original Parameter Estimator", "GRU", "LSTM", "DeepONet", "FNO"], loc="upper right", bbox_to_anchor=[1,6.5])
plt.savefig("trials.eps", bbox_inches="tight")

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

        outFNO = y_normalizer.decode(outFNO)
        outLSTM = y_normalizer.decode(outLSTM)
        outGRU = y_normalizer.decode(outGRU)

        testLossfno += loss(outFNO, y).item()
        testLosslstm += loss(outLSTM, y).item()
        testLossgru += loss(outGRU, y).item()

with torch.no_grad():
    testLossDeep = 0
    loss = LpLoss()
    for x, y in testDataNoNormal:
        x, y = x.cuda(), y.cuda()
        outDEEP = deepModel(x, grid_repeated)
        testLossdeep += loss(outDEEP, y).item()

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
    loss = LpLoss()
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()
        outGRU = gruModel(x)
        outLSTM = lstmModel(x)
        outFNO = fnoModel(x)

        outFNO = y_normalizer.decode(outFNO)
        outLSTM = y_normalizer.decode(outLSTM)
        outGRU = y_normalizer.decode(outGRU)
        y = y_normalizer.decode(y)

        testLossfno += loss(outFNO, y).item()
        testLosslstm += loss(outLSTM, y).item()
        testLossgru += loss(outGRU, y).item()

        testLossfno += loss(outFNO, y).item()
        testLosslstm += loss(outLSTM, y).item()
        testLossgru += loss(outGRU, y).item()

with torch.no_grad():
    testLossDeep = 0
    loss = LpLoss()
    for x, y in trainDataNoNormal:
        x, y = x.cuda(), y.cuda()
        outDEEP = deepModel(x, grid_repeated)
        testLossdeep += loss(outDEEP, y).item()

print("Total Relative Error Training Set FNO (900 Points)", testLossfno)
print("Total Relative Error Training Set GRU (900 Points)", testLossgru)
print("Total Relative Error Training Set LSTM (900 Points)", testLosslstm)
print("Total Relative Error Training Set DeepONet (900 Points)", testLossdeep)
print("Average Relative Error Training Set FNO (900 Points)", testLossfno/900)
print("Average Relative Error Training Set GRU (900 Points)", testLossgru/900)
print("Average Relative Error Training Set LSTM (900 Points)", testLosslstm/900)
print("Average Relative Error Training Set DeepONet (900 Points)", testLossdeep/900)
