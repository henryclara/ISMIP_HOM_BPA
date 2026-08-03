
from firedrake import *
import numpy as np
import os
from datetime import datetime

from config import *
from domain import *
from fields import *
from physics import *
from geometry import *
from spaces import *
from bcs import *

os.makedirs("Simulations", exist_ok=True)

Q1 = FunctionSpace(mesh3D, "CG", 1)
zeta = Function(Q1, name="zeta")
zeta_out = Function(Q1, name="zeta_out")

reset_state()

for dt in dts:
    simulation_start = datetime.now()
    num_TS = int(T / dt)
    outfile = VTKFile(f"Simulations/MISMIP2D_output_dt{dt:g}.pvd")

    uvec = Function(VV, name="uvec")
    uvec.assign(uvec_out)
    ux = split(uvec)[0]

    du = TrialFunction(VV)
    vvect = TestFunction(VV)
    v1 = split(vvect)[0]

    for i in range(num_TS):
        print(f"Step {i + 1}/{num_TS} at dt={dt:g}")

        mu = viscosity(ux, Constant(0.0), 3.0)
        F = (
            (4.0 * mu * ux.dx(0)) * v1.dx(0) * dx
            + (mu * ux.dx(1)) * v1.dx(1) * dx
            - rhoi * g * zs * v1.dx(0) * dx
        )

        phi_float = bed + (rhoi / rhow) * thick
        delta_GL = Constant(0.01)
        zeta.interpolate(0.5 * (1.0 - tanh(phi_float / delta_GL)))

        m = Constant(3.0)
        C = beta2 * sqrt(dot(uvec, uvec) + Constant(1.0e-10) ** 2) ** (1.0 / m - 1.0)
        F += (1.0 - zeta) * C * dot(uvec, vvect) * dx

        J = derivative(F, uvec, du)
        problem = NonlinearVariationalProblem(F, uvec, bcs=bcs_u, J=J)
        solver = NonlinearVariationalSolver(
            problem,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1.0e-5,
                "snes_atol": 1.0e-5,
                "snes_max_it": 50,
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

        solver.solve()
        uvec_out.assign(uvec)
        u_prev.assign(uvec_out)

        ubar = Function(VVbar, name="u_bar")
        ubar.project(uvec_out)
        ux_bar = split(ubar)[0]

        vel = as_vector([ux_bar])
        vnorm = sqrt(dot(vel, vel) + 1e-10)
        h = CellDiameter(mesh3D)
        mu_art = 0.1 * h * vnorm

        F = (
            thick_new * phi * dx
            - thick * phi * dx
            + dt * (ux_bar * thick_new).dx(0) * phi * dx
            - dt * (a_s - a_b) * phi * dx
            + dt * mu_art * dot(grad(thick_new), grad(phi)) * dx
        )

        solve(lhs(F) == rhs(F), H)
        thick.assign(H)
        thick.dat.data[:] = np.maximum(thick.dat.data, 10.0)

        zb.interpolate(max_value(bed, -rhoi / rhow * thick))
        zs.interpolate(zb + thick)

        zeta_out.interpolate(zeta)
        uout.interpolate(uvec_out)
        outfile.write(uout, thick, zs, zb, bed, zeta_out, time=(i + 1) * dt)

    simulation_end = datetime.now()
    print(f"Simulation time: {simulation_end - simulation_start}")
        