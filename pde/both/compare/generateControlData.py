import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy.linalg
from matplotlib.ticker import StrMethodFormatter
import sys
sys.path.append("../gru")
sys.path.append("../deeponet")
sys.path.append("../fno")
sys.path.append("../lstm")
from gru2 import GRUModel
from lstm2 import LSTMModel
from fourier import FNO1d
from deeponet import DeepONet
from deeponet import DeepONet2D
from normal import UnitGaussianNormalizer
import torch
import torch.nn as nn
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
from matplotlib import cm

# load our thetaHat array for the solution
timeLoop = []

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




# We need to get the thetaHat estimates for a few different cases and save them to be read by matlab
# First, load the data
L = 1
T = 14.95
dt = 0.05
dx = 0.02
nt = int(round(T/dt))
nx = int(round(L/dx))
npoints = 1000
batch_size = 20
t1 = np.linspace(0.05, T, nt, dtype=np.float32)
x1 = np.linspace(0.02, L, nx, dtype=np.float32)

x1 = np.linspace(0.02, L, nx, dtype=np.float32)
grid_repeated = []
grid_repeated2 = []
for i in range(batch_size):
    grid_repeated.append(t1)
    grid_repeated2.append(x1)
grid_repeated = torch.from_numpy(np.array(grid_repeated)).cuda()
grid_repeated2 = torch.from_numpy(np.array(grid_repeated2).reshape((batch_size, nx, 1))).cuda()

x1, t1 = np.meshgrid(x1, t1)

# Parameters
epochs = 500
ntrain = 900
ntest = 100
batch_size = 20
gamma = 0.5
learning_rate = 0.001
step_size= 100
modes = 12
width = 64
x1, t1 = np.meshgrid(x1, t1)

tHat = scipy.io.loadmat("thetaHat.mat")["tHatData"]
tHat = tHat.reshape(npoints, nt, nx).astype(np.float32)
xInit = scipy.io.loadmat("u.mat")["uData"]
xInit = xInit.reshape(npoints, nt, nx).astype(np.float32)
xArr = np.zeros((npoints, nt, 1))
for i in range(npoints):
    for j in range(nt):
        xArr[i][j][0] = xInit[i][j][0] 
xInit = xArr.astype(np.float32)
yGain = scipy.io.loadmat("gain.mat")["gainData"]
yGain = yGain.reshape(npoints, nt, nx).astype(np.float32)

# create a map to find out what the initial conditions were
xinit = np.zeros((npoints, 101))
for i in range(npoints):
    for j in range(101):
        xinit[i][j] = xInit[i, 0, 0]
delta = scipy.io.loadmat("delta.mat")["deltaData"]
# For some reason delta saved incorrectly, where everything but first col is 0
delta = delta.reshape((npoints, npoints))
delta = delta[:, 0]
# Create Mapping
mapping = {}
for i in range(npoints):
    mapping[tHat[i, 0, 0]] = (delta[i], xinit[i])

# Create train/test splits
x_train, x_test, y_train, y_test = train_test_split(xInit, tHat, test_size=0.1, random_state=1)
x_trainGain, x_testGain, y_trainGain, y_testGain = train_test_split(tHat, yGain, test_size=0.1, random_state=1)

x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()

x_trainGain = torch.from_numpy(x_trainGain).cuda()
y_trainGain = torch.from_numpy(y_trainGain).cuda()
x_testGain = torch.from_numpy(x_testGain).cuda()
y_testGain = torch.from_numpy(y_testGain).cuda()

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

trainDataGain = DataLoader(TensorDataset(x_trainGain, y_trainGain), batch_size=batch_size, shuffle=True)
testDataGain = DataLoader(TensorDataset(x_testGain, y_testGain), batch_size=batch_size, shuffle=False)

# Next, we load the models
fnoModel = FNO1d(modes, width, 1, 50).cuda()
gruModel = GRUModel(500, 1, 50).cuda()
lstmModel = LSTMModel(500, 1, 50).cuda()
branch = nn.Sequential(
        GRUModel(200, 1, 10*50)).cuda()
trunk = nn.Sequential(
        nn.Linear(1, 512),
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

with torch.no_grad():
    idx = 0
    for x, y in testData:
        x, y = x.cuda(), y.cuda()
        outFNO1 = fnoModel(x)
        outGRU1 = gruModel(x)
        outLSTM1 = lstmModel(x)
        outDEEP1 = deepModel(x, grid_repeated)
        idx+=1
        if idx == 3:
            break

# let us save estimations
deltaArr = []
xinit = []
etaArr = []
for i in range(batch_size):
    xinit.append(mapping[y[i, 0, 0].cpu().detach().numpy().item()][1])
    deltaArr.append(mapping[y[i, 0, 0].cpu().detach().numpy().item()][0])
deltaArr = np.array(deltaArr)
xinit = np.array(xinit)
etaArr = np.array(etaArr)
print(xinit)
print(x[0, 0, :])

scipy.io.savemat("outXInit.mat", dict(xinit=xinit))
scipy.io.savemat("outDelta.mat", dict(deltaArr=deltaArr))

# Modify output file to be extrapolated since we subsample
outFNO = outFNO1.detach().cpu().numpy()
outGRU = outGRU1.detach().cpu().numpy()
outLSTM = outLSTM1.detach().cpu().numpy()
outDEEP = outDEEP1.detach().cpu().numpy()
y = y.detach().cpu().numpy()
nt = 1501
nx = 101
thetaHatArrFNO = np.zeros((20, nt, nx))
thetaHatArrGRU = np.zeros((20, nt, nx))
thetaHatArrLSTM = np.zeros((20, nt, nx))
thetaHatArrDEEP = np.zeros((20, nt, nx))
yArr =np.zeros((20, nt, nx))
for i in range(batch_size):
    for j in range(nt):
        for k in range(nx):
            if j < 5:
                thetaHatArrFNO[i][j][k] = 0
                thetaHatArrGRU[i][j][k] = 0
                thetaHatArrLSTM[i][j][k] = 0
                thetaHatArrDEEP[i][j][k] = 0
                yArr[i][j][k] = 0
            else:
                if math.floor(j/5)-1 > outFNO.shape[1]-1:
                    thetaHatArrFNO[i][j][k] = outFNO[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    thetaHatArrGRU[i][j][k] = outGRU[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    thetaHatArrLSTM[i][j][k] = outLSTM[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    thetaHatArrDEEP[i][j][k] = outDEEP[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    yArr[i][j][k]  = y[i][math.floor(j/5)-2][math.floor(k/2)-1]
                else:
                    thetaHatArrFNO[i][j][k] = outFNO[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    thetaHatArrGRU[i][j][k] = outGRU[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    thetaHatArrLSTM[i][j][k] = outLSTM[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    thetaHatArrDEEP[i][j][k] = outDEEP[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    yArr[i][j][k] = y[i][math.floor(j/5)-1][math.floor(k/2)-1]


fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
t1 = np.linspace(0.05, T, nt, dtype=np.float32)
x1 = np.linspace(0.02, L, nx, dtype=np.float32)
x1, t1 = np.meshgrid(x1, t1)
axes[0].plot_surface(x1, t1, yArr[0, :, :])
axes[1].plot_surface(x1, t1, thetaHatArrDEEP[0, :, :])
plt.show()

scipy.io.savemat("estFNO.mat", dict(out=np.array(thetaHatArrFNO)))
scipy.io.savemat("estGRU.mat", dict(out=np.array(thetaHatArrGRU)))
scipy.io.savemat("estLSTM.mat", dict(out=np.array(thetaHatArrLSTM)))
scipy.io.savemat("estDEEP.mat", dict(out=np.array(thetaHatArrDEEP)))
scipy.io.savemat("estY.mat", dict(out=np.array(yArr)))


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

with torch.no_grad():
    idx = 0
    for x, y in testDataGain:
        x, y = x.cuda(), y.cuda()
        idx+=1
        if idx == 3:
            break

with torch.no_grad():
    outFNO2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()
    outGRU2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()
    outLSTM2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()
    outDEEP2 = torch.from_numpy(np.zeros((batch_size, x.shape[1], x.shape[2]), dtype=np.float32)).cuda()
    for timeStep in range(x.shape[1]):
        out = fnoModel2(outFNO1[:, timeStep, :].reshape(batch_size, x.shape[2], 1))
        outFNO2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])

        out = gruModel2(outGRU1[:, timeStep, :].reshape(batch_size, x.shape[2], 1))
        outGRU2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])

        out = lstmModel2(outLSTM1[:, timeStep, :].reshape(batch_size, x.shape[2], 1))
        outLSTM2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])

        out = deepModel2(outDEEP1[:, timeStep, :].reshape(batch_size, x.shape[2], 1), grid_repeated2)
        outDEEP2[:, timeStep, :] = out.reshape(batch_size, x.shape[2])

# Modify output file to be extrapolated since we subsample
outFNO = outFNO2.detach().cpu().numpy()
outGRU = outGRU2.detach().cpu().numpy()
outLSTM = outLSTM2.detach().cpu().numpy()
outDEEP = outDEEP2.detach().cpu().numpy()
y = y.detach().cpu().numpy()
nt = 1501
nx = 101
thetaHatArrFNO = np.zeros((20, nt, nx))
thetaHatArrGRU = np.zeros((20, nt, nx))
thetaHatArrLSTM = np.zeros((20, nt, nx))
thetaHatArrDEEP = np.zeros((20, nt, nx))
yArr =np.zeros((20, nt, nx))
for i in range(batch_size):
    for j in range(nt):
        for k in range(nx):
            if j < 5:
                thetaHatArrFNO[i][j][k] = 0
                thetaHatArrGRU[i][j][k] = 0
                thetaHatArrLSTM[i][j][k] = 0
                thetaHatArrDEEP[i][j][k] = 0
                yArr[i][j][k] = 0
            else:
                if math.floor(j/5)-1 > outFNO.shape[1]-1:
                    thetaHatArrFNO[i][j][k] = outFNO[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    thetaHatArrGRU[i][j][k] = outGRU[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    thetaHatArrLSTM[i][j][k] = outLSTM[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    thetaHatArrDEEP[i][j][k] = outDEEP[i][math.floor(j/5)-2][math.floor(k/2)-1]
                    yArr[i][j][k]  = y[i][math.floor(j/5)-2][math.floor(k/2)-1]
                else:
                    thetaHatArrFNO[i][j][k] = outFNO[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    thetaHatArrGRU[i][j][k] = outGRU[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    thetaHatArrLSTM[i][j][k] = outLSTM[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    thetaHatArrDEEP[i][j][k] = outDEEP[i][math.floor(j/5)-1][math.floor(k/2)-1]
                    yArr[i][j][k] = y[i][math.floor(j/5)-1][math.floor(k/2)-1]

fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
t1 = np.linspace(0.05, T, nt, dtype=np.float32)
x1 = np.linspace(0.02, L, nx, dtype=np.float32)
x1, t1 = np.meshgrid(x1, t1)
axes[0].plot_surface(x1, t1, yArr[0, :, :])
axes[1].plot_surface(x1, t1, thetaHatArrDEEP[0, :, :])
plt.show()

scipy.io.savemat("outFNO.mat", dict(out=np.array(thetaHatArrFNO)))
scipy.io.savemat("outGRU.mat", dict(out=np.array(thetaHatArrGRU)))
scipy.io.savemat("outLSTM.mat", dict(out=np.array(thetaHatArrLSTM)))
scipy.io.savemat("outDEEP.mat", dict(out=np.array(thetaHatArrDEEP)))
scipy.io.savemat("y.mat", dict(out=np.array(yArr)))

