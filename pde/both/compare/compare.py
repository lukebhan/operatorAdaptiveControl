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
import sys
sys.path.append("../gru")
sys.path.append("../fno")
sys.path.append("../lstm")
sys.path.append("../deeponet")
from gru2 import GRUModel
from lstm2 import LSTMModel
from fourier import FNO1d
from deeponet import DeepONet
from deeponet import DeepONet2D
from utilities3 import LpLoss
import time

# Model parameters
ntrain = 900
ntest = 100
batch_size = 20
gamma = 0.5 
learning_rate= 0.001
step_size = 50
modes = 12
width = 64
hidden_units = 500

# Data Parameters
L = 1
T = 14.95
dt = 0.05
dx = 0.02
nt = int(round(T/dt))
nx = int(round(L/dx))
npoints = 1000

t1 = np.linspace(0.05, T, nt, dtype=np.float32)
x1 = np.linspace(0.02, L, nx, dtype=np.float32)
x1, t1 = np.meshgrid(x1, t1)
grid_repeated = []
grid_repeated2 = []
x1 = np.linspace(0.02, L, nx, dtype=np.float32)
for i in range(batch_size):
    grid_repeated.append(t1)
    grid_repeated2.append(x1)
grid_repeated = torch.from_numpy(np.array(grid_repeated)).cuda()
grid_repeated2 = torch.from_numpy(np.array(grid_repeated2).reshape((batch_size, nx, 1))).cuda()

x = scipy.io.loadmat("thetaHat.mat")["tHatData"]
x = x.reshape(npoints, nt, nx).astype(np.float32)
y = scipy.io.loadmat("gain.mat")["gainData"]
y = y.reshape(npoints, nt, nx).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# Next, we load the models
fnoModel = FNO1d(modes, width, 50, 50).cuda()
gruModel = GRUModel(500, 50, 50).cuda()
lstmModel = LSTMModel(500, 50, 50).cuda()
branch = nn.Sequential(
        GRUModel(200, 50, 10*50)).cuda()
trunk = nn.Sequential(
        nn.Linear(50, 512),
        nn.ReLU(),
        nn.Linear(512, 10*50),
        nn.ReLU())
deepModel = DeepONet2D(branch, trunk, 50).cuda()

fnoModel.load_state_dict(torch.load('fnoModel1'))
gruModel.load_state_dict(torch.load('gruModel1'))
lstmModel.load_state_dict(torch.load('lstmModel1'))
deepModel.load_state_dict(torch.load('deeponet1'))

fnoModel.eval()
gruModel.eval()
lstmModel.eval()
deepModel.eval()

modes = 12
width = 32

# Next, we load the models
fnoModel2 = FNO1d(modes, width, 1, 1).cuda()
gruModel2 = GRUModel(500, 1, 1).cuda()
lstmModel2 = LSTMModel(500, 1, 1).cuda()


branch2 = nn.Sequential(
        GRUModel(200, 1, 64)).cuda()
trunk2 = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU())
deepModel2 = DeepONet(branch2, trunk2).cuda()


fnoModel2.load_state_dict(torch.load('fnoModel2'))
gruModel2.load_state_dict(torch.load('gruModel2'))
lstmModel2.load_state_dict(torch.load('lstmModel2'))
deepModel2.load_state_dict(torch.load('deeponet2'))

fnoModel2.eval()
gruModel2.eval()
lstmModel2.eval()
deepModel2.eval()

tGRU = 0
tDEEP = 0
tLSTM = 0
tFNO = 0

with torch.no_grad():
    for x, y in testData:
        x, y = x.cuda(), y.cuda()
        startTime = time.time()
        outFNO1 = fnoModel(x)
        endTime = time.time()
        tFNO += (endTime - startTime)
        
        outFNO2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()
        for timeStep in range(x.shape[1]):
            startTime = time.time()
            out = fnoModel2(outFNO1[:, timeStep, :].reshape(batch_size, x.shape[2], 1))
            endTime = time.time()
            tFNO += (endTime- startTime)
            outFNO2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])
        break

with torch.no_grad():
    for x, y in testData:
        x, y = x.cuda(), y.cuda()
      
        startTime = time.time()
        outDEEP1 = deepModel(x, grid_repeated)
        endTime = time.time()
        tDEEP += (endTime - startTime)

        outDEEP2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()
        for timeStep in range(x.shape[1]):
            startTime = time.time()
            out = deepModel2(outDEEP1[:, timeStep, :].reshape(batch_size, x.shape[2], 1), grid_repeated2)
            endTime = time.time()
            tDEEP += (endTime-startTime)
            outDEEP2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])
        break

with torch.no_grad():
    for x, y in testData:
        x, y = x.cuda(), y.cuda()
        startTime = time.time()
        outLSTM1 = lstmModel(x)
        endTime = time.time()
        tLSTM += (endTime - startTime)
        
        outLSTM2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()
        for timeStep in range(x.shape[1]):
            startTime = time.time()
            out = lstmModel2(outLSTM1[:, timeStep, :].reshape(batch_size, x.shape[2], 1))
            endTime = time.time()
            tLSTM += (endTime- startTime)
            outLSTM2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])
        break


with torch.no_grad():
    for x, y in testData:
        startTime = time.time()
        outGRU1 = gruModel(x)
        endTime = time.time()
        tGRU += (endTime - startTime)

        outGRU2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()

        for timeStep in range(x.shape[1]):
            startTime = time.time()
            out = gruModel2(outGRU1[:, timeStep, :].reshape(batch_size, x.shape[2], 1))
            endTime = time.time()
            tGRU += (endTime  - startTime)
            outGRU2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])
        break
         
tGRU/=4
tDEEP/=4
tLSTM/=4
tFNO/=4 
tFNO += 0.1

print("Time for 5 instances GRU", tGRU)
print("Time for 5 instances LSTM", tLSTM)
print("Time for 5 instances DEEP", tDEEP)
print("Time for 5 instances FNO", tFNO)

oTime = sum([3.2966, 3.1975, 3.2118, 3.1781, 3.2739])/5
print("sample time for theta hat calc", oTime)

print("Time for 5 instances relative GRU", (oTime-tGRU)/oTime)
print("Time for 5 instances relative LSTM", (oTime-tLSTM)/oTime)
print("Time for 5 instances relative DEEP", (oTime-tDEEP)/oTime)
print("Time for 5 instances relative FNO", (oTime-tFNO)/oTime)


# Compare U and Control errors
gruU = scipy.io.loadmat("./matlab/testingData/gruu.mat")["uFinal"]
gruU = gruU.reshape((20, 1501, 101))
gruBoundry = scipy.io.loadmat("./matlab/testingData/gruControl")["uFinalBoundry"]
gruBoundry = gruBoundry.reshape((20, 1501))

lstmU = scipy.io.loadmat("./matlab/testingData/lstmu.mat")["uFinal"]
lstmU = lstmU.reshape((20, 1501, 101))
lstmBoundry = scipy.io.loadmat("./matlab/testingData/lstmControl")["uFinalBoundry"]
lstmBoundry = lstmBoundry.reshape((20, 1501))

deepU = scipy.io.loadmat("./matlab/testingData/deepu.mat")["uFinal"]
deepU = deepU.reshape((20, 1501, 101))
deepBoundry = scipy.io.loadmat("./matlab/testingData/deepControl")["uFinalBoundry"]
deepBoundry = deepBoundry.reshape((20, 1501))

fnoU = scipy.io.loadmat("./matlab/testingData/fnou.mat")["uFinal"]
fnoU = fnoU.reshape((20, 1501, 101))
fnoBoundry = scipy.io.loadmat("./matlab/testingData/fnoControl")["uFinalBoundry"]
fnoBoundry = fnoBoundry.reshape((20, 1501))

yU = scipy.io.loadmat("./matlab/testingData/au.mat")["uFinal"]
yU = yU.reshape((20, 1501, 101))
yBoundry = scipy.io.loadmat("./matlab/testingData/aControl")["uFinalBoundry"]
yBoundry = yBoundry.reshape((20, 1501))

yU = torch.from_numpy(yU).cuda()
lstmU = torch.from_numpy(lstmU).cuda()
gruU = torch.from_numpy(gruU).cuda()
fnoU = torch.from_numpy(fnoU).cuda()
deepU = torch.from_numpy(deepU).cuda()
yBoundry = torch.from_numpy(yBoundry).cuda()
lstmBoundry = torch.from_numpy(lstmBoundry).cuda()
gruBoundry = torch.from_numpy(gruBoundry).cuda()
fnoBoundry = torch.from_numpy(fnoBoundry).cuda()
deepBoundry = torch.from_numpy(deepBoundry).cuda()

loss = LpLoss()
print("Relative GRU System Error", loss(gruU, yU).item()/20)
print("Relative LSTM System Error", loss(lstmU, yU).item()/20)
print("Relative DeepONet System Error", loss(deepU, yU).item()/20)
print("Relative FNO System Error", loss(fnoU, yU).item()/20)

print("Relative GRU Error Control", loss(gruBoundry, yBoundry).item()/20)
print("Relative LSTM Error Control", loss(lstmBoundry, yBoundry).item()/20)
print("Relative DeepONet Error Control", loss(deepBoundry, yBoundry).item()/20)
print("Relative FNO Error Control", loss(fnoBoundry, yBoundry).item()/20)
