
from firedrake import *
from netgen.occ import *
import numpy as np
import time

Lx = 80000.0
nz = 10

base = PeriodicIntervalMesh(50, Lx)
mesh = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)

x, sigma = SpatialCoordinate(mesh)

Xref = Function(mesh.coordinates.function_space(), name="Xref")
Xref.interpolate(SpatialCoordinate(mesh))
xref, sigmaref = split(Xref)

# Horizontal and vertical elements
horiz = FiniteElement("CG", interval, 1)
vert  = FiniteElement("CG", interval, 1)
scalar_elt = TensorProductElement(horiz, vert)
V = FunctionSpace(mesh, scalar_elt)

U = FunctionSpace(mesh, scalar_elt)

# Vector-valued version of the same tensor-product space
vector2_elt = VectorElement(scalar_elt, dim=2)
VV2 = FunctionSpace(mesh, vector2_elt)
uout = Function(VV2, name="uout")

Vbar = FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)
Ubar = FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)

W = U * Ubar * Vbar
w = Function(W)

u, u_s, q = split(w)
v, eta, r = TestFunctions(W)

u_out = Function(U, name="u")

phi = TestFunction(Vbar)
thick_new = TrialFunction(Vbar)
H = Function(Vbar)

u_prev = Function(U)
u_prev_ts = Function(U)

yearinsec = 365.25 * 24 * 60 * 60
A = Constant(1.0e-25 * yearinsec * 1.0e18)
#alpha = np.deg2rad(0.5)
omega = 2.0*np.pi / Lx
#tan_alpha = np.tan(alpha)
psi = np.deg2rad(1.0)
g = 9.8*yearinsec**2
rhoi = 917.0/(1.0e6*yearinsec**2)
rhow = 1028.0/(1.0e6*yearinsec**2)

zs = Function(Vbar, name="zs").interpolate(0.0)
zb = Function(Vbar, name="zb").interpolate(zs - 1000.0 \
             + 500.0 * sin(omega * x))

thick = Function(Vbar, name="thick").interpolate(zs - zb)

mesh.coordinates.interpolate(as_vector([xref, zb + sigmaref * thick]))
eps = Constant(1e-6) # Constant(1e-10)

def viscosity(u, n=3):
    eps_e2 = u.dx(0)**2 + 0.25 * u.dx(1)**2
    mu = 0.5 * A**(-1.0 / n) * (eps_e2 + eps**2)**((1.0 - n) / (2.0 * n))
    return mu

mu = 1
ns = [3] #np.linspace(1, 3, 11)

# Basal friction field
beta2 = Function(Vbar, name="beta2")
beta2.interpolate(1000.0 * (1.0 + sin(2.0*np.pi*x/Lx)))

a_s = 0.0
a_b = 0.0

dts = [50]
theta_outs = [1]

zeta = Constant(0.0)
T = 2000.0

for dt in dts:
    for theta_out in theta_outs:

        print("=" * 80)
        print(f"Starting run with dt={dt:g}, theta_out={theta_out:g}")
        print("=" * 80)

        theta = Constant(theta_out)
        num_TS = int(T / dt)

        zs.interpolate(0.0)
        zb.interpolate(zs - 1000.0 + 500.0 * sin(omega * x))
        thick.interpolate(zs - zb)
        mesh.coordinates.interpolate(as_vector([xref, zb + sigmaref * thick]))

        u_init = 10.0 * (1.0 + 0.01 * sin(2.0*pi*x/Lx))

        w.sub(0).interpolate(u_init)
        w.sub(1).interpolate(u_init)

        u0 = w.sub(0)
        w.sub(2).project(thick * u0.dx(0))

        u_prev.assign(0.0)
        u_prev_ts.assign(0.0)

        outfile = VTKFile(f"BPA_output_dt{dt:g}_theta{theta_out:g}_A1e-25_nx100_beta1000.pvd")

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
                    u, u_s, q = split(w)

                    dw = TrialFunction(W)
                    v, eta, r = TestFunctions(W)

                    u_for_form = u

                mu = viscosity(u_for_form, n)

                grad_zs_H = as_vector([zs.dx(0), zs.dx(1)])
                surf = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)

                mu = viscosity(u, n)

                F = (
                    4 * mu * u.dx(0) * v.dx(0) * dx
                    + mu * u.dx(1) * v.dx(1) * dx
                    + beta2 * u * v * ds_b
                )

                if theta_out != 0:
                    F += theta * rhoi * g * dt * (u_s * zs.dx(0)) * v.dx(0) * dx

                    F += theta * rhoi * g * np.cos(psi) * dt \
                        * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) \
                        * q * v.dx(0) * dx

                    F += 0.5 * theta * g * np.cos(psi) * dt \
                        * u * (zs * zs).dx(0) * v.dx(0) \
                        * (1 / sqrt(1 + zs.dx(0)**2)) * ds_t

                    F += theta * rhoi * g * np.cos(psi) * dt \
                        * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) \
                        * zs * (v.dx(0) / sqrt(1 + zs.dx(0)**2)) * q * ds_t

                    F += (u_s - u) * eta * ds_t
                    F += r * q * dx - r * thick * u.dx(0) * dx

                F -= rhoi * g * np.cos(psi) * zs * v.dx(0) * dx \
                    - rhoi * g * np.sin(psi) * v * dx

                if theta_out == 0:
                    J = derivative(F, uvec, du)
                    problem = NonlinearVariationalProblem(F, uvec, J=J)
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
                        "snes_max_it": 50,

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
                    uvec_out.assign(uvec)
                    w.sub(0).assign(uvec_out)
                else:
                    uvec_out.assign(w.sub(0))

                u_prev.assign(uvec_out)

            print("Solving thickness evolution now...")

            ubar = Function(VVbar, name="u_bar")
            ubar.project(u_out)

            vnorm = sqrt(ubar**2 + 1e-10)
            h = CellDiameter(mesh)
            mu_art = 0.1 * h * vnorm

            F = (
                thick_new * phi * dx
                - thick * phi * dx
                + dt * (ubar * thick_new).dx(0) * phi * dx
                + dt * mu_art * dot(grad(thick_new), grad(phi)) * dx
            )

            solve(lhs(F) == rhs(F), H)
            thick.assign(H)
            thick.dat.data[:] = np.maximum(thick.dat.data, 10.0)
            mesh.coordinates.interpolate(as_vector([xref, yref, zb + sigmaref * thick]))
            print("Finished solving thickness evolution...")
            print("Year: ", (i+1)*dt)

            t = (i + 1) * dt

            if abs(t % 50.0) < 1.0e-10 or abs((t % 50.0) - 50.0) < 1.0e-10:
                #uout.interpolate(as_vector([ux, uy, 0.0]))
                ux_out, uy_out = split(uvec_out)
                uout.interpolate(as_vector([ux_out, uy_out, 0.0]))
                outfile.write(uout, thick, time=t)
                