from firedrake import *

Lx = 640000.0
Lz = 1000.0

nx = 60
nz = 20

dts = [10.0]
theta_outs = [0]
FSSA_keyword = "none"
T = 20.0

yearinsec = 365.25 * 24 * 60 * 60
A = Constant(20.0)
g = 9.8 * yearinsec**2
rhoi = 917.0 / (1.0e6 * yearinsec**2)
rhow = 1028.0 / (1.0e6 * yearinsec**2)

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
zeta_pred = False
