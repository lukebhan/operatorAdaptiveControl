import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy.linalg
from matplotlib.ticker import FormatStrFormatter
import sys
sys.path.append("../gru")
sys.path.append("../deeponet")
sys.path.append("../fno")
sys.path.append("../lstm")
from gru2 import GRUModel
from lstm2 import LSTMModel
from fourier import FNO1d
from deeponet import DeepONet2D
from normal import UnitGaussianNormalizer
import torch
import torch.nn as nn
import time
import matplotlib
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

font = {'family': 'normal', 'weight':'bold', 'size':12}
matplotlib.rc('font', **font)


# holds time in solver
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






def thetaFunc(t):
    # map t to the index. 
    t *= 100
    t = np.round(t)
    if t >= 2000:
        return thetaHat[-1]
    else:
        return thetaHat[t]

# function for cpu computation
def func(y, t, theta, _, placeHolder):
    global timeLoop
    p = y[1]
    phi = y[0]
    rho = np.array([1, phi, p, np.abs(phi)*p, np.abs(p)*p]).reshape((5, 1))
    startTime = time.time()
    alpha = y[27:32].reshape((5, 1))

    intP = p*np.array([1, phi, p/2, np.abs(phi)*p/2, np.abs(p)*p/3]).reshape((5, 1))
    intDPhi = np.array([0, p, 0, p**2/2*np.sign(phi), 0]).reshape((5, 1))

    gamma = y[2:27].reshape((5, 5))

    # Calculate parameter values
    thetaHat = alpha + np.dot(gamma, intP)
    endTime = time.time()
    timeLoop.append(endTime - startTime)
    # Get control law
    constVec = np.array([-3/2, -2]).reshape((1, 2))
    u = (np.dot(constVec, np.array([y[0], y[1]]).reshape((2, 1))) - np.dot(rho.transpose(), thetaHat)).item()
    #u = np.sin(p)

    # Gamma update
    startTime = time.time()
    gammaDot = np.dot(-gamma, np.dot(rho, np.dot(rho.transpose(), gamma)))

    # Update alphadot
    firstTerm = np.dot(-gamma, np.dot(rho, np.dot(rho.transpose(), alpha)))
    secondTerm = np.dot(-gamma, np.dot(p, intDPhi))
    thirdTerm = np.dot(-gamma,  rho*u)
    alphaDot = firstTerm + secondTerm + thirdTerm
    endTime = time.time()
    timeLoop.append(endTime-startTime)

    # Solution
    xd1 = u + np.dot(rho.transpose(), theta).item()
    xdot  = np.array([p, xd1])

    xRet = np.concatenate((xdot, gammaDot.flatten(), alphaDot.flatten()))
    xRet = np.append(xRet, u)
    return xRet

def solveSystem(x0, theta, gamma, alpha, t, function=func, estimates=None, interpolate=None):
    x0 = np.concatenate((x0, gamma.flatten(), alpha.flatten()))
    x0 = np.append(x0, 0)
    sol = odeint(function, x0, t, args=(theta, estimates, interpolate))
    return sol

def getEstimates(sol, nt):
    theta = []
    gammaArr = []
    lypunov = []
    alphaRet = []
    controlArr = []
    for i in range(nt):
        alpha = sol[i][27:32].reshape((5, 1))
        gamma = sol[i][2:27].reshape((5, 5))
        control = sol[i][-1]

        p = sol[i][1]
        phi = sol[i][0]
        intP = p*np.array([1, phi, p/2, np.abs(phi)*p/2, np.abs(p)*p/3]).reshape((5, 1))
        res = alpha + np.dot(gamma, intP)

        lypunov.append(np.dot(res.transpose(), np.dot(np.linalg.inv(gamma), res)))
        theta.append(res)
        gammaArr.append(gamma)
        alpha = alpha.flatten()
        alphaRet.append(np.array([p, phi, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4]]))
        controlArr.append(control)
    theta = np.array(theta).reshape((nt, 5))
    theta = theta.transpose()
    alphaRet = np.array(alphaRet, dtype=np.float32)
    return theta, gammaArr, alphaRet, controlArr

def plotSol(sol):
    plt.figure()
    plt.plot(sol[0], sol[1])
    plt.title("Solution with Target Position [0, 0]")
    plt.xlabel(r"$\phi$")
    plt.ylabel("p")
    plt.savefig("sol.eps")
    plt.show()

def plotControl(sol):
    plt.figure(figsize=(6, 4))
    plt.title("Control")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.plot(t, sol[-1])
    plt.savefig("control.eps")
    plt.show()

def interpolateFunction(t, estimates):
    t = t*100
    if t <= 0:
        return estimates[0]
    elif t >= 2000:
        return estimates[-1]
    else:
        return estimates[int(t)]

def controlFunction(y, t, theta, estimates, interpolateFunc):
    # Get variables
    p = y[1]
    phi = y[0]
    rho = np.array([1, phi, p, np.abs(phi)*p, np.abs(p)*p]).reshape((5, 1))

    alpha = y[27:32].reshape((5, 1))

    intP = p*np.array([1, phi, p/2, np.abs(phi)*p/2, np.abs(p)*p/3]).reshape((5, 1))
    intDPhi = np.array([0, p, 0, p**2/2*np.sign(phi), 0]).reshape((5, 1))

    gamma = y[2:27].reshape((5, 5))

    # Calculate parameter values
    #thetaHat = alpha + np.dot(gamma, intP)
    thetaHat = interpolateFunc(t, estimates).reshape((5, 1))

    # Get control law
    constVec = np.array([-3/2, -2]).reshape((1, 2))
    u = (np.dot(constVec, np.array([y[0], y[1]]).reshape((2, 1))) - np.dot(rho.transpose(), thetaHat)).item()
    #u = np.sin(p)

    # Gamma update
    gammaDot = np.dot(-gamma, np.dot(rho, np.dot(rho.transpose(), gamma)))

    # Update alphadot
    firstTerm = np.dot(-gamma, np.dot(rho, np.dot(rho.transpose(), alpha)))
    secondTerm = np.dot(-gamma, np.dot(p, intDPhi))
    thirdTerm = np.dot(-gamma,  rho*u)
    alphaDot = firstTerm + secondTerm + thirdTerm

    # Solution
    xd1 = u + np.dot(rho.transpose(), theta).item()
    xdot  = np.array([p, xd1])

    xRet = np.concatenate((xdot, gammaDot.flatten(), alphaDot.flatten()))
    xRet = np.append(xRet, u)
    return xRet

def parseInputs(sol, testValue):
    # just gets the first element of the batch
    sol = x_normalizer.decode(sol)
    phi = sol[testValue, 0, 0].detach().cpu().numpy().item()
    p = sol[testValue, 0, 1].detach().cpu().numpy().item()
    alpha = sol[testValue, 0, 2:].detach().cpu().numpy()
    theta = thetasMap[phi]
    x0 = np.array([phi, p])
    alpha = alpha.reshape((5, 1))
    return x0, alpha, theta

def generateMap(x, thetas):
    # creates a map from x to thetas:
    mapping = {}
    for i in range(x.shape[0]):
        mapping[x[i, 0, 0]] = thetas[i]
    return mapping

T = 20
dt = 0.01
nt = int(round(T/dt))
t = np.linspace(0, T, nt, dtype=np.float32)
modes = 16
width = 48
hidden_units=850
batch_size = 20

# Load data so we have our normalizers
x = np.loadtxt("xAlpha.dat", dtype=np.float32)
x = x.reshape(
            x.shape[0], x.shape[1] // 7, 7)
y = np.loadtxt("y.dat", dtype=np.float32)
y = y.reshape(
            y.shape[0], y.shape[1] // 5, 5)
thetas = np.loadtxt("thetasArr.dat", dtype=np.float32)
thetasMap = generateMap(x, thetas)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cpu()
y_train = torch.from_numpy(y_train).cpu()
x_test = torch.from_numpy(x_test).cpu()
y_test = torch.from_numpy(y_test).cpu()
x_normalizer = UnitGaussianNormalizer(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
testDataNoNormal = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
x_test = x_normalizer.encode(x_test)
trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

grid_repeated = []
for i in range(batch_size):
    grid_repeated.append(t.reshape((2000, 1)))
grid_repeated = torch.from_numpy(np.array(grid_repeated)).cpu()

# Load Models
fnoModel = FNO1d(modes, width, 7, 5).cpu()
branch = nn.Sequential(GRUModel(200, 7, 500)).cpu()
trunk = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 500), nn.ReLU()).cpu()
deepModel = DeepONet2D(branch, trunk, 5).cpu()
gruModel = GRUModel(hidden_units, 7, 5).cpu()
lstmModel = LSTMModel(hidden_units, 7, 5).cpu()

deepModel.load_state_dict(torch.load('deeponet1'))
fnoModel.load_state_dict(torch.load('fnoModel1'))
gruModel.load_state_dict(torch.load('gruModel1'))
lstmModel.load_state_dict(torch.load('lstmModel1'))
fnoModel.eval()
deepModel.eval()
lstmModel.eval()
gruModel.eval()

# First get a solution

with torch.no_grad():
    for x, y in testDataNoNormal:
        x, y = x.cpu(), y.cpu()
        deepStartTime = time.time()
        deepOut = deepModel(x, grid_repeated.cpu())
        deepEndTime = time.time()

with torch.no_grad():
    for x, y in testData:
        x, y = x.cpu(), y.cpu()
        gruStartTime = time.time()
        gruOut = gruModel(x)
        gruEstimates = y_normalizer.decode(gruOut)
        gruEndTime = time.time()

        fnoStartTime = time.time()
        fnoOut = fnoModel(x)
        fnoEstimates = y_normalizer.decode(fnoOut)
        fnoEndTime = time.time()


        lstmStartTime = time.time()
        lstmOut = lstmModel(x)
        lstmEstimates = y_normalizer.decode(lstmOut)
        lstmEndTime = time.time()

fig, ax = plt.subplots(2, 4, figsize=(15, 4))
fig.subplots_adjust(hspace=0.5, wspace=0.3)
shift = 1
times = 0
for k in range(4):
    # This gets our original x0, alpha, and thetaValues
    x0, alpha, theta = parseInputs(x, k+shift)
    gamma = np.identity(5)
    # sol with control
    sol = solveSystem(x0, theta, gamma, alpha, t)
    times += sum(timeLoop)
    timeLoop = []

    # subtract time for other calculation of system
    thetaTrue = theta.copy()
    theta, _, _, _ = getEstimates(sol, nt)

    #for i in range(5):
        #plt.figure()
        #plt.plot(t, theta[i], label="Original Estimator Theta")
        #plt.plot(t, fnoEstimates[testValue, :, i].cpu().detach(), label="FNO Estimator Theta")
        #plt.legend()
        #plt.show()

    # Now we apply those estimates to our control
    estimatesFNO = fnoEstimates[k+shift, :, :].cpu().detach().numpy().reshape((2000, 5))
    estimatesDeep = deepOut[k+shift, :, :].cpu().detach().numpy().reshape((2000, 5))
    estimatesLSTM = lstmEstimates[k+shift, :, :].cpu().detach().numpy().reshape((2000, 5))
    estimatesGRU = gruEstimates[k+shift, :, :].cpu().detach().numpy().reshape((2000, 5))

    solControlFNO = solveSystem(x0, thetaTrue, gamma, alpha, t, controlFunction, estimatesFNO, interpolateFunction)
    solControlDeep = solveSystem(x0, thetaTrue, gamma, alpha, t, controlFunction, estimatesDeep, interpolateFunction)
    solControlLSTM = solveSystem(x0, thetaTrue, gamma, alpha, t, controlFunction, estimatesLSTM, interpolateFunction)
    solControlGRU = solveSystem(x0, thetaTrue, gamma, alpha, t, controlFunction, estimatesGRU, interpolateFunction)

    sol = sol.transpose()
    solControlFNO = solControlFNO.transpose()
    solControlGRU = solControlGRU.transpose()
    solControlLSTM = solControlLSTM.transpose()
    solControlDeep = solControlDeep.transpose()

    #ax[1][k].plot(t, solControlGRU[-1], label="GRU", linestyle=linestyle_tuple[1][1])
    ax[1][k].plot(t, sol[-1], label="Original Parameter Estimator")
    ax[1][k].plot(t, solControlGRU[-1], label="GRU", linestyle=linestyle_tuple[1][1])
    ax[1][k].plot(t, solControlLSTM[-1], label="LSTM", linestyle=linestyle_tuple[3][1])
    ax[1][k].plot(t, solControlDeep[-1], label="DeepONet", linestyle=linestyle_tuple[5][1])
    ax[1][k].plot(t, solControlFNO[-1], label="FNO", linestyle=linestyle_tuple[4][1])
    ax[1][k].set_xlabel("t", font)
    ax[1][0].set_ylabel("u", font)

    axins = ax[1][k].inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.plot(t, sol[-1], label="Original Parameter Estimator")
    axins.plot(t, solControlGRU[-1], label="GRU", linestyle=linestyle_tuple[1][1])
    axins.plot(t, solControlLSTM[-1], label="LSTM", linestyle=linestyle_tuple[3][1])
    axins.plot(t, solControlDeep[-1], label="DeepONet", linestyle=linestyle_tuple[5][1])
    axins.plot(t, solControlFNO[-1], label="FNO", linestyle=linestyle_tuple[4][1])

    axins.set_xlim(17, 18)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.set_xticks([], minor=True)
    axins.set_yticks([], minor=True)
    axins.tick_params(bottom=False, left=False)
    if k==0:
        axins.set_ylim(sol[-1][1699]-0.1, sol[-1][1799]+0.1)
    elif k==1:
        axins.set_ylim(sol[-1][1699]-0.5, sol[-1][1799]+0.5)
    elif k==2:
        axins.set_ylim(sol[-1][1699]-0.5, sol[-1][1799]+0.5)
    elif k==3:
        axins.set_ylim(sol[-1][1699]-0.3, sol[-1][1799]+0.3)
    elif k==4:
        axins.set_ylim(sol[-1][1699]-0.5, sol[-1][1799]+0.8)
    ax[1][k].indicate_inset_zoom(axins, alpha=1)





    ax[0][k].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0][k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    #l1, = ax[0][k].plot(solControlGRU[0], solControlGRU[1], label="GRU", linestyle=linestyle_tuple[1][1])
    l1, = ax[0][k].plot(sol[0], sol[1], label="Orginal Parameter Estimator")
    l2, = ax[0][k].plot(solControlGRU[0], solControlGRU[1], label="GRU", linestyle=linestyle_tuple[1][1])
    l3, = ax[0][k].plot(solControlLSTM[0], solControlLSTM[1], label="LSTM", linestyle=linestyle_tuple[3][1])
    l4, = ax[0][k].plot(solControlDeep[0], solControlDeep[1], label="Deep", linestyle=linestyle_tuple[5][1])
    l5, = ax[0][k].plot(solControlFNO[0], solControlFNO[1], label="FNO", linestyle =linestyle_tuple[4][1] )

    ax[0][k].set_title("Example " + str(k+1), font)
    ax[0][k].set_xlabel(r"$\phi$", font)
    ax[0][0].set_ylabel("p", font)
plt.legend([l1, l2, l3, l4, l5], ["Original Estimator", "GRU", "LSTM", "DeepONet", "FNO"],loc="lower left", ncol=5, bbox_to_anchor=[-3, -.7])
plt.savefig("controls.eps", bbox_inches="tight")

print("Time to calculate Original Estimator for 5 Examples", times)
# batch size = 20 
print("Time to calculate GRU for 5 Examples", (gruEndTime-gruStartTime)/4, "% Increase:", (times-((gruEndTime-gruStartTime)/4))/times)
print("Time to calculate LSTM for 5 Examples", (lstmEndTime-lstmStartTime)/4, "% Increase:", (times-((lstmEndTime-lstmStartTime)/4))/times)
print("Time to calculate DeepONet for 5 Examples", (deepEndTime-deepStartTime)/4, "% Increase:", (times-((deepEndTime-deepStartTime)/4) )/times)
print("Time to calculate FNO for 5 Examples", (fnoEndTime-fnoStartTime)/4, "% Increase:", (times-((fnoEndTime-fnoStartTime)/4))/times)
print("Time for original estimator", times)
