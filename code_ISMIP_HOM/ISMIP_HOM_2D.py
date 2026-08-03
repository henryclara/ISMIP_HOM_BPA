
from math import sqrt

from firedrake import *
from firedrake import CheckpointFile
import numpy as np
import os
import csv
from datetime import datetime

def save_restart(filename, mesh, w, u_out, thick, zs, zb, t, dt, nx, nz, time_stepping):
    """
    Save the current simulation state for later error calculations or restart.

    Current 2D variables:
        w.sub(0) = u
        w.sub(1) = q
        u_out    = scalar horizontal velocity
        thick    = ice thickness
        zs       = surface elevation
        zb       = bed elevation
    """
    with CheckpointFile(filename, "w") as afile:
        afile.save_mesh(mesh)

        afile.save_function(w, name="w")
        afile.save_function(u_out, name="u")
        afile.save_function(thick, name="thick")
        afile.save_function(zs, name="zs")
        afile.save_function(zb, name="zb")

        # Optional metadata as attributes
        afile.h5pyfile.attrs["time"] = float(t)
        afile.h5pyfile.attrs["dt"] = float(dt)
        afile.h5pyfile.attrs["nx"] = int(nx)
        afile.h5pyfile.attrs["nz"] = int(nz)
        afile.h5pyfile.attrs["time_stepping"] = time_stepping

Lx = 80000.0
nx = 3200
nz = 320

base = PeriodicIntervalMesh(nx, Lx)
mesh = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)

x, sigma = SpatialCoordinate(mesh)

Xref = Function(mesh.coordinates.function_space(), name="Xref")
Xref.interpolate(SpatialCoordinate(mesh))
xref, sigmaref = split(Xref)

# Horizontal and vertical elements
horiz = FiniteElement("CG", interval, 1)
vert  = FiniteElement("CG", interval, 1)
scalar_elt = TensorProductElement(horiz, vert)

U = FunctionSpace(mesh, scalar_elt)

# Vector-valued version of the same tensor-product space
vector2_elt = VectorElement(scalar_elt, dim=2)
VV2 = FunctionSpace(mesh, vector2_elt)
uout = Function(VV2, name="uout")

Vbar = FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)

W = U * Vbar
w = Function(W)

u, q = split(w)
v, r = TestFunctions(W)

u_out = Function(U, name="u")

phi = TestFunction(Vbar)
thick_new = TrialFunction(Vbar)
H = Function(Vbar)

u_prev = Function(U)
u_prev_ts = Function(U)

yearinsec = 365.25 * 24 * 60 * 60
A = Constant(2.0e-25 * yearinsec * 1.0e18)
#alpha = np.deg2rad(0.5)
omega = 2.0*np.pi / Lx
#tan_alpha = np.tan(alpha)
psi = np.deg2rad(2.0)
g = 9.8*yearinsec**2
rhoi = 917.0/(1.0e6*yearinsec**2)
rhow = 1028.0/(1.0e6*yearinsec**2)

eps = Constant(1e-6) # Constant(1e-10)

def viscosity(u, n=3):
    eps_e2 = u.dx(0)**2 + 0.25 * u.dx(1)**2
    mu = 0.5 * A**(-1.0 / n) * (eps_e2 + eps**2)**((1.0 - n) / (2.0 * n))
    return mu

mu = 1
ns = [3]

# Basal friction field
beta2 = Function(Vbar, name="beta2")
beta2.interpolate(1000.0 * (1.0 + sin(2.0*pi*xref/Lx)))

a_s = Constant(0.0)
a_b = Constant(0.0)

dts = [0.1, 1, 2, 5, 10, 20, 50]
theta_outs = [0.5]

zeta = Constant(0.0)
T = 100
time_stepping = "im_mi"

for dt in dts:
    for theta_out in theta_outs:

        if dt == 0.01 and theta_out == 1:
            continue

        print("=" * 80)
        print(f"Starting run with dt={dt:g}, theta_out={theta_out:g}")
        print("=" * 80)

        simulation_start = datetime.now()

        theta = Constant(theta_out)
        num_TS = int(T / dt)

        zs = Function(Vbar, name="zs").interpolate(500.0 * sin(omega * xref))
        zb = Function(Vbar, name="zb").interpolate(- 1000.0 + 500.0 * sin(omega * xref))
        thick = Function(Vbar, name="thick").interpolate(zs - zb)
        mesh.coordinates.interpolate(as_vector([xref, zb + sigmaref * thick]))

        u_init = 10.0 * (1.0 + 0.01 * sin(2.0*pi*xref/Lx))

        w.sub(0).interpolate(u_init)

        u0 = w.sub(0)
        w.sub(1).project(thick * u0.dx(0))

        u_prev.assign(0.0)
        u_prev_ts.assign(0.0)

        outfile = VTKFile(f"Simulations/BPA_output_dt{dt:g}_theta{theta_out:g}_nx{nx}_nz{nz}.pvd")

        for i in range(num_TS):
            for j, n in enumerate(ns):
                print("Solving with n = ", n)
                change=100
                tol=1e-3
                maxiter=200
                iter_sim=0

                #while change>tol and iter_sim<maxiter:
                #iter_sim=iter_sim+1

                if theta_out == 0:
                    u_solve = Function(U)
                    u_solve.interpolate(w.sub(0))

                    du = TrialFunction(U)
                    v = TestFunction(U)

                    u_for_form = u_solve

                else:
                    u, q = split(w)

                    dw = TrialFunction(W)
                    v, r = TestFunctions(W)

                    u_for_form = u

                mu = viscosity(u_for_form, n)

                F = (
                    4 * mu * u_for_form.dx(0) * v.dx(0) * dx
                    + mu * u_for_form.dx(1) * v.dx(1) * dx
                    + beta2 * u_for_form * v * ds_b
                )

                nz_s = 1.0 / sqrt(1.0 + zs.dx(0)**2)
                nz_b = 1.0 / sqrt(1.0 + zb.dx(0)**2)

                if theta_out != 0:

                    F += theta * rhoi * g * cos(psi) * dt * q * v.dx(0) * dx
                    F -= theta * rhoi * g * cos(psi) * dt * a_s * v.dx(0) * dx
                    
                    F += theta * rhoi * g * cos(psi) * dt * nz_s * q * v * zs.dx(0) * ds_t
                    F -= theta * rhoi * g * cos(psi) * dt * nz_s * a_s * v * zs.dx(0) * ds_t

                    F += theta * rhoi * g * cos(psi) * dt * nz_b * q * v * zb.dx(0) * ds_b
                    F -= theta * rhoi * g * cos(psi) * dt * nz_b * a_s * v * zb.dx(0) * ds_b

                    # Constraint for q
                    F += r * q * dx - r * thick * u.dx(0) * dx

                F -= rhoi * g * cos(psi) * zs * v.dx(0) * dx
                F += rhoi * g * sin(psi) * v * dx
                
                F -= rhoi * g * cos(psi) * nz_s * zs * v * zs.dx(0) * ds_t
                F -= rhoi * g * cos(psi) * nz_b * zs * v * zb.dx(0) * ds_b

                if theta_out == 0:
                    J = derivative(F, u_solve, du)
                    problem = NonlinearVariationalProblem(F, u_solve, J=J)
                else:
                    J = derivative(F, w, dw)
                    problem = NonlinearVariationalProblem(F, w, J=J)

                solver = NonlinearVariationalSolver(
                    problem,
                    solver_parameters={
                        "snes_type": "newtonls",
                        "snes_linesearch_type": "bt",
                        "snes_rtol": 1.0e-8,
                        "snes_atol": 1.0e-10,
                        "snes_max_it": 100,

                        "mat_type": "aij",
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type": "mumps",

                        "snes_monitor": None,
                        "ksp_monitor_true_residual": None,
                        "ksp_error_if_not_converged": None,
                    },
                )

                solver.solve()
                if theta_out == 0:
                    u_out.assign(u_solve)
                    w.sub(0).assign(u_out)
                else:
                    u_out.assign(w.sub(0))

                u_prev.assign(u_out)

            print("Solving thickness evolution now...")

            ubar = Function(Vbar, name="u_bar")
            ubar.project(u_out)

            vnorm = sqrt(ubar**2 + 1e-10)
            h = CellDiameter(mesh)
            mu_art = 0.1 * h * vnorm

            if time_stepping == "im":
                F = (
                    thick_new * phi * dx
                    - thick * phi * dx
                    + dt * (ubar * thick_new).dx(0) * phi * dx
                    + dt * mu_art * thick_new.dx(0) * phi.dx(0) * dx
                )
            if time_stepping == "im_mi":
                H_mid = 0.5 * (thick_new + thick)

                F = (
                    thick_new * phi * dx
                    - thick * phi * dx
                    + dt * (ubar * H_mid).dx(0) * phi * dx
                    + dt * mu_art * H_mid.dx(0) * phi.dx(0) * dx
                )

            solve(lhs(F) == rhs(F), H)

            thick.assign(H)
            thick.dat.data[:] = np.maximum(thick.dat.data, 10.0)
            zs.assign(zb + thick)
            mesh.coordinates.interpolate(as_vector([xref, zb + sigmaref * thick]))

            print("Finished solving thickness evolution...")
            print("Year: ", (i+1)*dt)

            t = (i + 1) * dt

            if abs(t % 10) < 1.0e-10 or abs((t % 10) - 10) < 1.0e-10:
                #uout.interpolate(as_vector([ux, uy, 0.0]))
                uout.interpolate(as_vector([u_out, 0.0]))
                outfile.write(uout, thick, zs, zb, time=t)
                restart_dir = f"Simulations/BPA_output_dt{dt:g}_theta{theta_out:g}_nx{nx}_nz{nz}"
                os.makedirs(restart_dir, exist_ok=True)
                restart_file = f"{restart_dir}/restart_t{t:g}.h5"
                save_restart(restart_file,mesh,w,u_out,thick,zs,zb,t,dt,nx,nz,time_stepping)

        simulation_end = datetime.now()
        os.makedirs("Simulations", exist_ok=True)
        with open("Simulations/simulation_times.txt", "a") as f:
            f.write(f"dt={dt:g}, theta_out={theta_out:g}, nx={nx}, nz={nz}, "
                f"T={T:g}, start={simulation_start}, end={simulation_end}, "
                f"runtime={simulation_end - simulation_start}\n"
            )
