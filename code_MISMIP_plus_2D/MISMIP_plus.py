
from firedrake import *
import numpy as np
import os
from config import *
from datetime import datetime

start_step = 0
restart_from = None #f"Simulations/MISMIP_output_theta1_dt10/restart_t{start_step}.h5"
if restart_from is not None:
    os.environ["RESTART_MESH_FILE"] = restart_from

from domain import *
from fields import *
from physics import *
from geometry import *
from spaces import *
from bcs import *
from io_local import *

Q1 = FunctionSpace(mesh3D, "CG", 1)
zeta = Function(Q1, name="zeta")
zeta_out = Function(Q1, name="zeta_out")
zeta_predicted = Function(Q1, name="zeta_predicted")

for dt in dts:
    for theta_out in theta_outs:

        simulation_start = datetime.now()

        if restart_from is None:
            reset_state()
            t_restart = 0.0
            start_step = 0
        else:
            t_restart, start_step, dt_restart, theta_restart = load_restart(restart_from)
            u_prev.assign(uvec_out)
            print(f"Restarting from t={t_restart:g}, step={start_step}")
        
        print("=" * 80)
        print(f"Starting run with dt={dt:g}, theta_out={theta_out:g}")
        print("=" * 80)

        theta = Constant(theta_out)
        num_TS = int(T / dt)

        outfile = VTKFile(f"Simulations/MISMIP_output_theta{theta_out:g}_dt{dt:g}_GL_pred{zeta_pred}.pvd")

        if theta_out == 0:
            u = Function(V)
            u.assign(w.sub(0))
            u_s = u
            u_b = u
            q = Function(Vbar, name="q")
            q.assign(0.0)

            du = TrialFunction(V)
            v = TestFunction(V)

        else:
            u, u_s, u_b, q = split(w)

            dw = TrialFunction(W)
            v, eta, xi, r = TestFunctions(W)

        for i in range(start_step, num_TS):
            print("Solving momentum")

            mu = viscosity(u, 3.0)

            surf = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)

            F = (4 * mu * u.dx(0)) * v.dx(0) * dx + mu * u.dx(1) * v.dx(1) * dx

            phi_float = bed + (rhoi/rhow) * thick
            delta_GL = Constant(0.01)
            grounded_prediction = Function(Q1, name="grounded_prediction")

            n = FacetNormal(mesh3D)

            F -= rhoi * g * zs * v.dx(0) * dx

            if zeta_pred == False:
                zeta.interpolate(0.5 * (1.0 - tanh(phi_float / delta_GL)))

            elif zeta_pred == True:
                H_k_plus_1 = thick - dt * (q + (u * zs.dx(0)) - (u * zb.dx(0)) - a_s + a_b)
                phi_pred = bed + (rhoi/rhow) * H_k_plus_1
                zeta.interpolate(0.5 * (1.0 - tanh(phi_pred / delta_GL)))
                
            zeta_predicted.assign(zeta)

            m = Constant(3.0)
            C = beta2 * sqrt(u**2 + Constant(1.0e-10)**2) ** (1.0/m - 1.0)
            F += (1 - zeta) * C * u * v * ds_b

            if theta_out != 0 and FSSA_keyword == "full":
                
                # First line (excluding first term)
                F += theta * rhoi * g * dt * ((1 - zeta * rhoi/rhow) * (u_s * zs.dx(0) - u_b * zb.dx(0) + q - a_s) - (1 + zeta * rhoi/rhow) * a_b) * v.dx(0) * dx

                # Second line
                F += theta * rhoi * g * dt * n[2] * zs * ((1 - zeta * (rhoi/rhow)) * (u_s * zs.dx(0) - u_b * zb.dx(0) + q) + zeta * (rhoi/rhow) * (a_s - a_b) - a_b) * v.dx(0) * ds_t

                # Third line
                F += theta * rhoi * g * dt * n[2] * zs * (- zeta * (rhoi/rhow) * ((u_s * zs.dx(0)) - (u_b * zb.dx(0)) + q) + zeta * (rhoi/rhow) * (a_s - a_b) - a_b) * v.dx(0) * ds_b

                # Fourth line
                F -= theta * dt * rhoi * g * a_s * n[2] * (zs.dx(0) * v) * ds_t
                F += theta * zeta * dt * rhoi * g * a_b * n[2] * (zs.dx(0) * v) * ds_b

                # The constraints:
                F += (u_s - u) * eta * ds_t
                F += (u_b - u) * xi * ds_b
                F += r * q * dx - r * thick * u.dx(0) * dx

            if theta_out == 0:
                J = derivative(F, u, du)
                problem = NonlinearVariationalProblem(F, u, bcs=bcs_u, J=J)
            else:
                J = derivative(F, w, dw)
                problem = NonlinearVariationalProblem(F, w, bcs=bcs_w, J=J)

            solver = NonlinearVariationalSolver(
                problem,
                solver_parameters={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1.0e-5,
                    "snes_atol": 1.0e-5,
                    "snes_max_it": 200,

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
                uvec_out.assign(u)
                w.sub(0).assign(uvec_out)
            else:
                uvec_out.assign(w.sub(0))

            u_prev.assign(uvec_out)

            print("Solving thickness evolution now...")

            ubar = Function(Vbar, name="u_bar")
            ubar.project(uvec_out)
            ux_bar = ubar

            vnorm = sqrt(ux_bar**2 + 1e-10)
            h = CellDiameter(mesh3D)
            mu_art = 0.1 * h * vnorm

            F = (
                thick_new * phi * dx \
                - thick * phi * dx \
                + dt * (ux_bar * thick_new).dx(0) * phi * dx
                - dt * (a_s - a_b) * phi * dx
                # Artifical viscosity
                + dt * mu_art * dot(grad(thick_new), grad(phi)) * dx
            )

            old_thick = Function(Vbar)
            old_thick.assign(thick)

            solve(lhs(F) == rhs(F), H)
            thick.assign(H)
            thick.dat.data[:] = np.maximum(thick.dat.data, 10.0)

            z_ref = Function(Q1, name="z_ref")
            z_ref.interpolate(SpatialCoordinate(mesh3D)[2])

            sigma_ref = Function(Q1, name="sigma_ref")
            sigma_ref.interpolate((z_ref - zb) / thick)

            #sigma = sigmaref

            zb_float = -rhoi / rhow * thick
            zb.interpolate(max_value(bed, zb_float))
            zs.interpolate(zb + thick)

            phi_float_actual = bed + (rhoi/rhow) * thick
            zeta.interpolate(0.5 * (1.0 - tanh(phi_float_actual / delta_GL)))

            mesh3D.coordinates.interpolate(as_vector([xref, yref, zb + sigma_ref * thick]))
            print("Finished solving thickness evolution...")
            

            if restart_from is None:
                print("Year: ", (i+1)*dt)
            else:
                print("Year: ", (t_restart + (i - start_step + 1) * dt))

            if restart_from is None:
                t = (i + 1) * dt
            else:
                t = t_restart + (i - start_step + 1) * dt

            if abs(t % dt) < 1.0e-10 or abs((t % dt) - dt) < 1.0e-10:
                ux_out = uvec_out
                ux_s = u_s
                ux_b = u_b
                uout.interpolate(as_vector([ux_out, 0.0]))
                usout.interpolate(as_vector([ux_s, 0.0]))
                ubout.interpolate(as_vector([ux_b, 0.0]))
                zeta_out.interpolate(zeta)
                outfile.write(uout, usout, ubout, thick, zs, zb, bed, zeta_out, zeta_predicted, time=t)
                restart_dir = f"Simulations/MISMIP_output_theta{theta_out:g}_dt{dt:g}_GL_pred{zeta_pred}"
                os.makedirs(restart_dir, exist_ok=True)
                restart_file = f"{restart_dir}/restart_t{t:g}.h5"
                save_restart(restart_file, t, i+1, dt, theta_out)

        simulation_end = datetime.now()
        print(f"Simulation time: {simulation_end - simulation_start}")
