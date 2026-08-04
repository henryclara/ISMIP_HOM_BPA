from firedrake import *
import numpy as np

Lx = 640000.0
Ly_full = 80000.0
Ly = Ly_full / 2.0

nx = int(640/4)
ny = int(40/4)
nz = int(10/2)

dts = [10]
theta_outs = [1]
FSSA_keyword = "full"
T = 10000.0

yearinsec = 365.25 * 24 * 60 * 60
A = Constant(20.0)
#omega = 2.0*np.pi / Lx
g = 9.8*yearinsec**2
rhoi = 917.0/(1.0e6*yearinsec**2)
rhow = 1028.0/(1.0e6*yearinsec**2)

xbar = Constant(300000.0)
B0 = Constant(-150.0)
B2 = Constant(-728.8)
B4 = Constant(343.91)
B6 = Constant(-50.57)

wc = Constant(24000.0)
fc = Constant(4000.0)
dc = Constant(500.0)
zdeep = Constant(-720.0)

a_s = Constant(0.3)
a_b = Constant(0.0)

n = 3.0

eps_H = 100.0
zeta_pred = True
