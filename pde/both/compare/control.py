import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import numpy.linalg
from matplotlib.ticker import FormatStrFormatter
import sys
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

t1 = np.linspace(0.05, T, nt)
x1 = np.linspace(0.02, L, nx)

# Parameters
epochs = 500
ntrain = 900
ntest = 100
batch_size = 20
gamma = 0.5
learning_rate = 0.001
step_size= 100
modes = 12
width = 32


x = scipy.io.loadmat("u.mat")["uData"]
x = x.reshape(npoints, nt, nx, 1).astype(np.float32)
y = scipy.io.loadmat("thetaHat.mat")["tHatData"]
y = y.reshape(npoints, nt, nx, 1).astype(np.float32)
# create a map to find out what the initial conditions were
xinit = scipy.io.loadmat("uInitData.mat")["uInitData"]
# nx=101 because we dont subsample
xinit = xinit.reshape(npoints, 101)
delta = scipy.io.loadmat("delta.mat")["deltaData"]
# For some reason delta saved incorrectly, where everything but first col is 0
delta = delta.reshape((npoints, npoints))
delta = delta[:, 0]

# Create train/test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for x, y in testData:
        x, y = x.cuda(), y.cuda()

fnoControlArr = np.zeros((5, 1501))
lstmControlArr = np.zeros((5, 1501))
gruControlArr = np.zeros((5, 1501))
origControlArr = np.zeros((5, 1501))
deepControlArr = np.zeros((5, 1501))
fnoSystemArr = np.zeros((5, 1501, 101))
gruSystemArr = np.zeros((5, 1501, 101))
lstmSystemArr = np.zeros((5, 1501, 101))
origSystemArr = np.zeros((5, 1501, 101))
deepSystemArr = np.zeros((5, 1501, 101))
fnoSystemArrNoDiff = np.zeros((5, 1501, 101))
gruSystemArrNoDiff = np.zeros((5, 1501, 101))
lstmSystemArrNoDiff = np.zeros((5, 1501, 101))
origSystemArrNoDiff = np.zeros((5, 1501, 101))
deepSystemArrNoDiff = np.zeros((5, 1501, 101))


for i in range(1):
    origControl = scipy.io.loadmat("./matlab/sample" + str(i+1) +"/aControl.mat")["uBoundry"]
    origControl = origControl.reshape(1501)
    origSystem  = scipy.io.loadmat("./matlab/sample" + str(i+1) +"/au.mat")["u"]
    origSystem = origSystem.reshape(1501, 101)
    origControlArr[i] = origControl
    origSystemArrNoDiff[i] = origSystem

    fnoControl = scipy.io.loadmat("./matlab/sample" + str(i+1) +"/fnoControl.mat")["uBoundry"]
    fnoControl = fnoControl.reshape(1501)
    fnoSystem  = scipy.io.loadmat("./matlab/sample" + str(i+1) +"/fnou.mat")["u"]
    fnoSystem = fnoSystem.reshape(1501, 101)
    fnoControlArr[i] = fnoControl
    fnoSystemArrNoDiff[i] = fnoSystem

    gruControl = scipy.io.loadmat("./matlab/sample" + str(i+1)+"/gruControl.mat")["uBoundry"]
    gruControl = gruControl.reshape(1501)
    gruSystem  = scipy.io.loadmat("./matlab/sample" + str(i+1) + "/gruu.mat")["u"]
    gruSystem = gruSystem.reshape(1501, 101)
    gruControlArr[i] = gruControl
    gruSystemArrNoDiff[i] = gruSystem

    lstmControl = scipy.io.loadmat("./matlab/sample" + str(i+1)+ "/lstmControl.mat")["uBoundry"]
    lstmControl = lstmControl.reshape(1501)
    lstmSystem  = scipy.io.loadmat("./matlab/sample" + str(i+1) + "/lstmu.mat")["u"]
    lstmSystem = lstmSystem.reshape(1501, 101)
    lstmControlArr[i] = lstmControl
    lstmSystemArrNoDiff[i] = lstmSystem

    deepControl = scipy.io.loadmat("./matlab/sample" + str(i+1)+ "/deepControl.mat")["uBoundry"]
    deepControl = deepControl.reshape(1501)
    deepSystem  = scipy.io.loadmat("./matlab/sample" + str(i+1) + "/deepu.mat")["u"]
    deepSystem = deepSystem.reshape(1501, 101)
    deepControlArr[i] = deepControl
    deepSystemArrNoDiff[i] = deepSystem

#  downsample
x1 = np.linspace(0.02, 1, 50)
t1 = np.linspace(0.05, 19.95, 299)
x1, t1 = np.meshgrid(x1, t1)
vmin = np.ones(5)*100000.0
vmax = np.zeros(5)
i= 0
fnoSystemDown = np.zeros((5, 299, 50))
lstmSystemDown = np.zeros((5, 299, 50))
gruSystemDown = np.zeros((5, 299, 50))
deepSystemDown = np.zeros((5, 299, 50))

fnoSystemDownNoDiff = np.zeros((5, 299, 50))
lstmSystemDownNoDiff = np.zeros((5, 299, 50))
gruSystemDownNoDiff = np.zeros((5, 299, 50))
deepSystemDownNoDiff = np.zeros((5, 299, 50))
origSystemDownNoDiff = np.zeros((5, 299, 50))
# downsample
for i in range(5):
    for j in range(299):
        for k in range(50):
            fnoSystemDown[i][j][k] = math.sqrt((fnoSystemArrNoDiff[i][j*5][k*2] - origSystemArrNoDiff[i][j*5][k*2])**2)
            lstmSystemDown[i][j][k] = math.sqrt((lstmSystemArrNoDiff[i][j*5][k*2] - origSystemArrNoDiff[i][j*5][k*2])**2)
            gruSystemDown[i][j][k] = math.sqrt((gruSystemArrNoDiff[i][j*5][k*2]- origSystemArrNoDiff[i][j*5][k*2])**2)
            deepSystemDown[i][j][k] = math.sqrt((deepSystemArrNoDiff[i][j*5][k*2]- origSystemArrNoDiff[i][j*5][k*2])**2)

            fnoSystemDownNoDiff[i][j][k] = fnoSystemArrNoDiff[i][j*5][k*2]
            lstmSystemDownNoDiff[i][j][k] = lstmSystemArrNoDiff[i][j*5][k*2]
            gruSystemDownNoDiff[i][j][k] = gruSystemArrNoDiff[i][j*5][k*2]
            deepSystemDownNoDiff[i][j][k] = deepSystemArrNoDiff[i][j*5][k*2]
            origSystemDownNoDiff[i][j][k] = origSystemArrNoDiff[i][j*5][k*2]

font = {'family': 'normal', 'weight':'bold', 'size':12}
fontSmall = {'family': 'normal', 'weight':'bold', 'size':5}
matplotlib.rc('font', **font)

# plot u
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
vmin1 = 10000
vmax1=0
sampleVal = 0
vmin1 = min(vmin1, np.min(deepSystemDownNoDiff[sampleVal, :, :]))
vmin1 = min(vmin1, np.min(fnoSystemDownNoDiff[sampleVal, :, :]))
vmin1 = min(vmin1, np.min(gruSystemDownNoDiff[sampleVal, :, :]))
vmin1 = min(vmin1, np.min(lstmSystemDownNoDiff[sampleVal, :, :]))
vmin1 = min(vmin1, np.min(origSystemDownNoDiff[sampleVal, :, :]))
vmax1 = max(vmax1, np.max(origSystemDownNoDiff[sampleVal, :, :]))
vmax1 = max(vmax1, np.max(gruSystemDownNoDiff[sampleVal, :, :]))
vmax1 = max(vmax1, np.max(lstmSystemDownNoDiff[sampleVal, :, :]))
vmax1 = max(vmax1, np.max(deepSystemDownNoDiff[sampleVal, :, :]))
vmax1 = max(vmax1, np.max(fnoSystemDownNoDiff[sampleVal, :, :]))
 
pad = 5 # in points
axes[0].imshow(origSystemDownNoDiff[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin1, vmax=vmax1)
axes[1].imshow(gruSystemDownNoDiff[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin1, vmax=vmax1)
axes[2].imshow(lstmSystemDownNoDiff[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin1, vmax=vmax1)
axes[3].imshow(deepSystemDownNoDiff[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin1, vmax=vmax1)
res = axes[4].imshow(fnoSystemDownNoDiff[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin1, vmax=vmax1)
axes[0].set_title("Original Gain", font)
axes[1].set_title("GRU", font)
axes[2].set_title("LSTM", font)
axes[3].set_title("DeepONet", font)
axes[4].set_title("FNO", font)
axes[0].set_xlabel("t", font)
axes[1].set_xlabel("t", font)
axes[2].set_xlabel("t", font)
axes[3].set_xlabel("t", font)
axes[4].set_xlabel("t", font)

plt.colorbar(res, ax=axes.ravel().tolist(), pad=0.01)
axes[0].set_ylabel("x", font)
plt.savefig("control1.eps", bbox_inches="tight")

# plot u error
vmin2 = 10000
vmax2=0
vmin2 = min(vmin2, np.min(np.sqrt((gruControlArr[sampleVal]-origControlArr[sampleVal])**2)))
vmin2 = min(vmin2, np.min(np.sqrt((lstmControlArr[sampleVal]-origControlArr[sampleVal])**2)))
vmin2 = min(vmin2, np.min(np.sqrt((deepControlArr[sampleVal]-origControlArr[sampleVal])**2)))
vmin2 = min(vmin2, np.min(np.sqrt((fnoControlArr[sampleVal]-origControlArr[sampleVal])**2)))
vmax2 = max(vmax2, np.max(np.sqrt((gruControlArr[sampleVal]-origControlArr[sampleVal])**2)))
vmax2 = max(vmax2, np.max(np.sqrt((lstmControlArr[sampleVal]-origControlArr[sampleVal])**2)))
vmax2 = max(vmax2, np.max(np.sqrt((deepControlArr[sampleVal]-origControlArr[sampleVal])**2)))
vmax2 = max(vmax2, np.max(np.sqrt((fnoControlArr[sampleVal]-origControlArr[sampleVal])**2)))


# plot control error
fig, axes = plt.subplots(1, 4, figsize=(20, 3))
tTicks = np.linspace(0, 20, 1501)
axes[0].set_ylim([vmin2, vmax2])
axes[1].set_ylim([vmin2, vmax2])
axes[2].set_ylim([vmin2, vmax2])
axes[3].set_ylim([vmin2, vmax2])
print(gruControlArr[sampleVal] - origControlArr[sampleVal])
for i in range(1501):
    print(gruControlArr[sampleVal][i] - origControlArr[sampleVal][i])
l2 = axes[0].plot(tTicks, np.sqrt((gruControlArr[sampleVal]-origControlArr[sampleVal])**2), label="GRU", linestyle=linestyle_tuple[3][1])
l3 = axes[1].plot(tTicks, np.sqrt((lstmControlArr[sampleVal]-origControlArr[sampleVal])**2), label="LSTM", linestyle=linestyle_tuple[3][1])
l4 = axes[2].plot(tTicks, np.sqrt((deepControlArr[sampleVal]-origControlArr[sampleVal])**2), label="DeepONet", linestyle=linestyle_tuple[3][1])
l5 = axes[3].plot(tTicks, np.sqrt((fnoControlArr[sampleVal]-origControlArr[sampleVal])**2), label="FNO", linestyle=linestyle_tuple[3][1])
axes[0].set_title("GRU", font)
axes[1].set_title("LSTM", font)
axes[2].set_title("DeepONet", font)
axes[3].set_title("FNO", font)
axes[0].set_ylabel(r"Control $L_2$ Error", font)
axes[0].set_xlabel("t", font)
axes[1].set_xlabel("t", font)
axes[2].set_xlabel("t", font)
axes[3].set_xlabel("t", font)
plt.savefig("control3.eps", bbox_inches="tight")

# plot control in one fig
fig, axes = plt.subplots(1, 4, figsize=(20, 3))
t = np.linspace(0, 15, 1501)
axes[0].plot(t, origControlArr[sampleVal], label="Original Estimator")
axes[1].plot(t, origControlArr[sampleVal], label="Original Estimator")
axes[2].plot(t, origControlArr[sampleVal], label="Original Estimator")
axes[3].plot(t, origControlArr[sampleVal], label="Original Estimator")

axes[0].plot(t, gruControlArr[sampleVal], label="GRU", linestyle=linestyle_tuple[1][1])
axes[1].plot(t, lstmControlArr[sampleVal], label="LSTM", linestyle=linestyle_tuple[1][1])
axes[2].plot(t, deepControlArr[sampleVal], label="DeepONet", linestyle=linestyle_tuple[1][1])
axes[3].plot(t, fnoControlArr[sampleVal], label="FNO", linestyle=linestyle_tuple[1][1])

axes[0].set_title("GRU", font)
axes[1].set_title("LSTM", font)
axes[2].set_title("DeepONet", font)
axes[3].set_title("FNO", font)
axes[0].set_xlabel("t", font)
axes[1].set_xlabel("t", font)
axes[2].set_xlabel("t", font)
axes[3].set_xlabel("t", font)
axes[0].set_ylabel("U(t)", font)
axins = axes[0].inset_axes([0.79, 0.03, 0.2, 0.2])
axins.plot(t, origControlArr[sampleVal], label="Original Estimator")
axins.plot(t, gruControlArr[sampleVal], label="GRU", linestyle=linestyle_tuple[1][1])
axins.set_xlim(13.2, 13.3)
axins.set_ylim(-3, 0)
axins.set_xticklabels([])
axins.set_xticks([], minor=True)
axins.set_yticks([-3, 0])
axins.yaxis.set_tick_params(labelsize=8)
axes[0].indicate_inset_zoom(axins, alpha=1)

axins = axes[1].inset_axes([0.79, 0.03, 0.2, 0.2])
axins.plot(t, origControlArr[sampleVal], label="Original Estimator")
axins.plot(t, lstmControlArr[sampleVal], label="LSTM", linestyle=linestyle_tuple[1][1])
axins.set_xlim(13.2, 13.3)
axins.set_ylim(-3, 0)
axins.set_xticklabels([])
axins.set_xticks([], minor=True)
axins.set_yticks([-3, 0])
axins.yaxis.set_tick_params(labelsize=8)
axes[1].indicate_inset_zoom(axins, alpha=1)


axins = axes[2].inset_axes([0.79, 0.03, 0.2, 0.2])
axins.plot(t, origControlArr[sampleVal], label="Original Estimator")
axins.plot(t, deepControlArr[sampleVal], label="LSTM", linestyle=linestyle_tuple[1][1])
axins.set_xlim(13.2, 13.3)
axins.set_ylim(-3, 0)
axins.set_xticklabels([])
axins.set_xticks([], minor=True)
axins.set_yticks([-3, 0])
axins.yaxis.set_tick_params(labelsize=8)
axes[2].indicate_inset_zoom(axins, alpha=1)


axins = axes[3].inset_axes([0.79, 0.03, 0.2, 0.2])
axins.plot(t, origControlArr[sampleVal], label="Original Gain")
axins.plot(t, fnoControlArr[sampleVal], label="LSTM", linestyle=linestyle_tuple[1][1])
axins.set_xlim(13.2, 13.3)
axins.set_ylim(-3, 0)
axins.set_xticklabels([])
axins.set_xticks([], minor=True)
axins.set_yticks([-3, 0])
axins.yaxis.set_tick_params(labelsize=8)
axes[3].indicate_inset_zoom(axins, alpha=1)






fig.savefig("control5.eps", bbox_inches="tight")




# plut u error
for i in range(5):
    vmin[i] = min(vmin[i], np.min(lstmSystemDown[i]))
    vmin[i] = min(vmin[i], np.min(gruSystemDown[i]))
    vmin[i] = min(vmin[i], np.min(fnoSystemDown[i]))
    vmin[i] = min(vmin[i], np.min(deepSystemDown[i]))
    vmax[i] = max(vmax[i], np.max(fnoSystemDown[i]))
    vmax[i] = max(vmax[i], np.max(lstmSystemDown[i]))
    vmax[i] = max(vmax[i], np.max(gruSystemDown[i]))
    vmax[i] = max(vmax[i], np.max(deepSystemDown[i]))



fig, axes = plt.subplots(1,4, figsize=(20, 3))
axes[0].imshow(gruSystemDown[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin[sampleVal], vmax=vmax[sampleVal])
axes[1].imshow(lstmSystemDown[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin[sampleVal], vmax=vmax[sampleVal])
axes[2].imshow(deepSystemDown[sampleVal, :, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin[sampleVal], vmax=vmax[sampleVal])
res = axes[3].imshow(fnoSystemDown[sampleVal,:, :].transpose(), interpolation='bicubic', aspect='auto', origin='lower', extent=[0, 20, 0, 1], vmin=vmin[sampleVal], vmax=vmax[sampleVal])
plt.colorbar(res, ax=axes.ravel().tolist(), pad=0.01)
axes[0].set_xlabel("t", font)
axes[1].set_xlabel("t", font)
axes[2].set_xlabel("t", font)
axes[3].set_xlabel("t", font)
axes[0].set_title("GRU", font)
axes[1].set_title("LSTM", font)
axes[2].set_title("DeepONet", font)
axes[3].set_title("FNO", font)

plt.savefig("control2.eps", bbox_inches="tight")

gruSumError = np.zeros(299)
lstmSumError = np.zeros(299)
fnoSumError = np.zeros(299)
deepSumError = np.zeros(299)
for j in range(299):
    gruSumError[j] = sum(gruSystemDown[sampleVal, j, :])
    lstmSumError[j] = sum(lstmSystemDown[sampleVal, j, :])
    fnoSumError[j] = sum(fnoSystemDown[sampleVal, j, :])
    deepSumError[j] = sum(deepSystemDown[sampleVal, j, :])


vmin3 = 10000
vmax3=0
vmin3 = min(vmin3, np.min(gruSumError))
vmin3 = min(vmin3, np.min(lstmSumError))
vmin3 = min(vmin3, np.min(deepSumError))
vmin3 = min(vmin3, np.min(fnoSumError))
vmax3 = max(vmax3, np.max(gruSumError))
vmax3 = max(vmax3, np.max(lstmSumError))
vmax3 = max(vmax3, np.max(deepSumError))
vmax3 = max(vmax3, np.max(fnoSumError))


# plot error over x at time t for u
fig, axes = plt.subplots(1, 4, figsize=(20, 3))
tTicks = np.linspace(0, 20, 299)
axes[0].set_ylim([vmin3, vmax3])
axes[1].set_ylim([vmin3, vmax3])
axes[2].set_ylim([vmin3, vmax3])
axes[3].set_ylim([vmin3, vmax3])
l2 = axes[0].plot(tTicks, gruSumError, label="GRU", linestyle=linestyle_tuple[1][1])
l3 = axes[1].plot(tTicks, lstmSumError, label="LSTM", linestyle=linestyle_tuple[1][1])
l4 = axes[2].plot(tTicks, deepSumError, label="DeepONet", linestyle=linestyle_tuple[1][1])
l5 = axes[3].plot(tTicks, fnoSumError, label="FNO", linestyle=linestyle_tuple[1][1])
axes[0].set_title("GRU", font)
axes[1].set_title("LSTM", font)
axes[2].set_title("DeepONet", font)
axes[3].set_title("FNO", font)
axes[0].set_ylabel(r"Summed $L_2$ Error u at Each Time", font)
axes[0].set_xlabel("t", font)
axes[1].set_xlabel("t", font)
axes[2].set_xlabel("t", font)
axes[3].set_xlabel("t", font)

plt.savefig("control4.eps", bbox_inches="tight")



