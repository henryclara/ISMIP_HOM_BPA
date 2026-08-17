from firedrake import *
import numpy as np

Lx = 640000.0
Ly_full = 80000.0
Ly = Ly_full / 2.0
nz = int(10)

dts = [20, 10, 5, 2, 1, 0.1]
theta_outs = [1]
FSSA_keyword = "full"
zeta_pred = False
time_stepping = "im_mi"
T = 10200
output_int = 100
coarse_res = 2000.0
refined_res = 1000.0
exp_name = "Ice1rr"  # "Ice0", "Ice1rr", "Ice1ra"
restart_from = f"Simulations/remesh_mesh_res_{str(int(coarse_res))}_{str(int(refined_res))}_test/restart_t10000.h5"

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
