from firedrake import *
import numpy as np

Lx = 640000.0
Ly_full = 80000.0
Ly = Ly_full / 2.0
nz = int(10)

dts = [0.5]
theta_outs = [1]
FSSA_keyword = "full"
zeta_pred = False
time_stepping = "im"
t = 0
T = 8000
output_int = 100
coarse_res = 2000.0
refined_res = 2000.0
exp_name = "Ice0"  # "Ice0", "Ice1rr", "Ice1ra"
tau_gjp = Constant(0.1)
restart_from = None #f"Simulations/remesh_mesh_res_2000_2000_unstructured/restart_t5000.h5"
# restart_from = f"Simulations/Ice1rr_theta{str(theta_outs[0])}_dt{str(dts[0])}_res{str(int(coarse_res))}_{str(int(refined_res))}_nz10/restart_t10100.h5"

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
