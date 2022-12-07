import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def func(y, t, theta, _):
    x = y[0]
    xdot = theta*x
    alpha = y[1]
    alphadot = -x**2 * (x**2 + alpha)
    return np.array((xdot, alphadot))

def plotSystem(sol):
    plt.figure()
    sol = np.array(sol).transpose()
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.plot(sol[0])
    plt.savefig("sol.eps")

def getEstimates(sol):
    sol = np.array(sol).transpose()
    thetaArr = []
    for i in range(len(sol[0])):
        theta = 0.5*(sol[0][i]**2 + sol[1][i])
        thetaArr.append(theta)
    return np.array(thetaArr)

def plotTheta(sol):
    plt.figure()
    plt.xlabel("t")
    plt.ylabel(r"$\hat{\theta}$")
    plt.plot(getEstimates(sol))
    plt.savefig("theta.eps")

def solveSystem(t, x0, alpha0, theta):
    y0 = np.array([x0, alpha0])
    sol = odeint(func, y0, t, args=((theta, theta)))
    return sol

def solveRandomSystem(t):
    theta = np.random.uniform(0, 5)
    x0 = np.random.uniform(0, 10)
    alpha0 = np.random.uniform(0, 10)
    y0 = np.array([x0, alpha0])
    sol = odeint(func, y0, t, args=((theta, theta)))
    return sol, theta, alpha0

def makeDataArrs(sol, theta, alpha0):
    thetas = getEstimates(sol)
    x_data = []
    y_data = thetas
    for i in range(len(thetas)):
        x_data.append(np.array([sol[i][0], alpha0]))
    return np.array(x_data), y_data

T = 1
nt = 100
t = np.linspace(0, T, nt)

x = []
y = []
for i in range(1):
    sol = solveSystem(t, 1, 1, 3)
    plotTheta(sol)
    plotSystem(sol)
