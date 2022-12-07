import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import scipy.io

L = 1
T = 14.95
dt = 0.05
dx = 0.02 
nt = int(round(T/dt))
nx = int(round(L/dx))
npoints = 1000

t1 = np.linspace(dt, T, nt)
x1 = np.linspace(dx, L, nx)

font = {'family': 'normal', 'weight':'bold', 'size':12}
matplotlib.rc('font', **font)

x = scipy.io.loadmat("u.mat")["uData"]
x = x.reshape(npoints, nt, nx, 1).astype(np.float32)
y = scipy.io.loadmat("thetaHat.mat")["tHatData"]
y = y.reshape(npoints, nt, nx, 1).astype(np.float32)
gain = scipy.io.loadmat("gain.mat")["gainData"]
gain = gain.reshape(npoints, nt, nx, 1).astype(np.float32)
delta= scipy.io.loadmat("delta.mat")["deltaData"]
delta = delta.reshape(npoints, npoints)
delta = delta.transpose()[0]

x1, t1 = np.meshgrid(x1, t1)
fig1, ax1 = plt.subplots(2, 5, subplot_kw={"projection":"3d"}, figsize=(25, 10))
fig2, ax2 = plt.subplots(2, 5, subplot_kw={"projection":"3d"}, figsize=(25, 10))
fig3, ax3 = plt.subplots(2, 5, subplot_kw={"projection":"3d"}, figsize=(25, 10))
fig1.suptitle(r"$\hat{\theta}(x, t)$", fontweight='bold', fontsize='20', y=0.95)
fig2.suptitle("u(x, t)", fontweight='bold', fontsize='20', y=0.92)
fig3.suptitle(r"$\kappa(x, t)$", fontweight='bold', fontsize='20', y=0.95)
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig3.subplots_adjust(wspace=0.2)
fig1.subplots_adjust(wspace=0.1)
fig2.subplots_adjust(wspace=0.1)
fmt = lambda x: "{:.2f}".format(x)
for i in range(10):
    inst = i
    print(x[inst, 0, 0, 0])

    ax1[int(i/5)][i%5].tick_params(axis="z", pad=3)
    surf = ax1[int(i/5)][i%5].plot_surface(x1, t1, y[inst, :, :, 0], cmap="YlGnBu_r")
    ax1[int(i/5)][i%5].set_xlabel("x", font)
    ax1[int(i/5)][i%5].set_ylabel("t", font)
    ax1[int(i/5)][i%5].set_zlabel(r"$\hat{\theta}(x, t)$", font, labelpad=5)

    surf = ax2[int(i/5)][i%5].plot_surface(x1, t1, x[inst, :, :, 0], cmap="YlGnBu_r")
    ax2[int(i/5)][i%5].tick_params(axis="z", pad=10)
    ax2[int(i/5)][i%5].set_xlabel("x", font)
    ax2[int(i/5)][i%5].set_ylabel("t", font)
    ax2[int(i/5)][i%5].set_zlabel("u(x, t)", font,labelpad=15)

    ax2[int(i/5)][i%5].view_init(20, 10)

    surf = ax3[int(i/5)][i%5].plot_surface(x1, t1, gain[inst, :, :, 0], cmap="YlGnBu_r")
    ax2[int(i/5)][i%5].tick_params(axis="z", pad=7)
    ax3[int(i/5)][i%5].set_xlabel("x", font)
    ax3[int(i/5)][i%5].set_ylabel("t", font)
    ax3[int(i/5)][i%5].set_zlabel(r"$\kappa(x, t)$", font, labelpad=10)
    ax3[int(i/5)][i%5].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax3[int(i/5)][i%5].set_yticks([0, 2, 4, 6, 8, 10, 12 , 14])

fig1.savefig("theta1.eps", bbox_inches="tight")
fig2.savefig("u1.eps", bbox_inches="tight")
fig3.savefig("gain1.eps", bbox_inches="tight")
