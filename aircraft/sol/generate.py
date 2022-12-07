import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy.linalg

def func(y, t, theta, _):
    # Get variables
    p = y[1]
    phi = y[0]
    rho = np.array([1, phi, p, np.abs(phi)*p, np.abs(p)*p]).reshape((5, 1))

    alpha = y[27:32].reshape((5, 1))

    intP = p*np.array([1, phi, p/2, np.abs(phi)*p/2, np.abs(p)*p/3]).reshape((5, 1))
    intDPhi = np.array([0, p, 0, p**2/2*np.sign(phi), 0]).reshape((5, 1))

    gamma = y[2:27].reshape((5, 5))

    # Calculate parameter values
    thetaHat = alpha + np.dot(gamma, intP)

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

T = 20
dt = 0.01
nt = int(round(T/dt))
t = np.linspace(0, T, nt)

X = []
Y = []
Xalpha = []
thetasArr = []
for i in range(1000):
    if i % 100 == 0:
        print(i)

    #theta = np.array([0, -26.67, 0.76485, -2.9225, 0]).reshape((5, 1))
    theta = np.array([np.random.uniform(0, 1), np.random.uniform(-24, -28), np.random.uniform(.5, 1), np.random.uniform(-2, -4), np.random.uniform(-0.01, 0)]).reshape((5, 1))
    thetasArr.append(theta)

    x0 = np.array([np.random.uniform(0, 1), 0.0])
    gamma = np.identity(5)
    alpha = theta.copy()*np.random.uniform(0, 2)

    x0 = np.concatenate((x0, gamma.flatten(), alpha.flatten()))
    # add u
    x0 = np.append(x0, 0)

    sol = odeint(func, x0, t, args=(theta, theta))
    theta = []
    gammaArr = []
    lypunov = []
    xArr = []
    xAlphaArr = []
    for i in range(nt):
        alpha = sol[i][27:32].reshape((5, 1))
        gamma = sol[i][2:27].reshape((5, 5))

        p = sol[i][1]
        phi = sol[i][0]
        intP = p*np.array([1, phi, p/2, np.abs(phi)*p/2, np.abs(p)*p/3]).reshape((5, 1))
        res = alpha + np.dot(gamma, intP)

        lypunov.append(np.dot(res.transpose(), np.dot(np.linalg.inv(gamma), res)))
        theta.append(res)
        gammaArr.append(gamma)
        xArr.append([p, phi])
        alpha = alpha.reshape(5)
        xAlphaArr.append([phi, p, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4]])

    sol = sol.transpose()
    theta = np.array(theta).reshape((nt, 5))
    gamma = np.array(gammaArr).reshape((nt, 25))
    gamma = gamma.transpose()
    lypunov = np.array(lypunov).reshape((nt))
    xArr = np.array(xArr).reshape((nt, 2))
    xAlphaArr = np.array(xAlphaArr).reshape((nt, 7))

    X.append(xArr)
    Y.append(theta)
    Xalpha.append(xAlphaArr)

X = np.array(X)
Y = np.array(Y)
Xalpha = np.array(Xalpha)
thetasArr = np.array(thetasArr)

print(X.shape)
print(Y.shape)
print(Xalpha.shape)
print(thetasArr.shape)

x_reshaped = X.reshape(X.shape[0], -1)
y_reshaped = Y.reshape(Y.shape[0], -1)
Xalpha_reshaped = Xalpha.reshape(Xalpha.shape[0], -1)
thetasArr_reshaped=  thetasArr.reshape(thetasArr.shape[0], -1)

np.savetxt("x.dat", x_reshaped)
np.savetxt("y.dat", y_reshaped)
np.savetxt("xAlpha.dat", Xalpha_reshaped)
np.savetxt("thetasArr.dat", thetasArr_reshaped)

