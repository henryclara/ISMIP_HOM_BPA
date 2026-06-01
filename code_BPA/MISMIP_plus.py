from math import sqrt, tanh
from pyclbr import Function
from turtle import dot

from firedrake import *
import numpy as np
import os
from config import *

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
#grounded_prediction_out = Function(Q1, name="grounded_prediction_out")

for dt in dts:
    for theta_out in theta_outs:

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
            uvec = Function(VV)
            uvec.assign(w.sub(0))

            ux, uy = split(uvec)

            du = TrialFunction(VV)
            vvect = TestFunction(VV)
            v1, v2 = split(vvect)

        else:
            uvec, u_s, u_b, q = split(w)
            ux, uy = split(uvec)
            ux_s, uy_s = split(u_s)
            ux_b, uy_b = split(u_b)

            dw = TrialFunction(W)
            vvect, eta, xi, r = TestFunctions(W)
            v1, v2 = split(vvect)

        for i in range(start_step, num_TS):
            print("Solving momentum")

            mu = viscosity(ux, uy, 3.0)

            surf = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)

            F = (4 * mu * ux.dx(0) + 2 * mu * uy.dx(1)) * v1.dx(0) * dx \
                            + (mu * ux.dx(1) + mu * uy.dx(0)) * v1.dx(1) * dx \
                            + mu * ux.dx(2) * v1.dx(2) * dx
                        
            F += (4 * mu * uy.dx(1) + 2 * mu * ux.dx(0)) * v2.dx(1) * dx \
                            + (mu * uy.dx(0) + mu * ux.dx(1)) * v2.dx(0) * dx \
                            + mu * uy.dx(2) * v2.dx(2) * dx

            phi_float = bed + (rhoi/rhow) * thick
            delta_GL = Constant(50.0)
            grounded_prediction = Function(Q1, name="grounded_prediction")

            n = FacetNormal(mesh3D)
        

            F -= rhoi * g * zs * (v1.dx(0) + v2.dx(1)) * dx

            if zeta_pred == False:
                zeta.interpolate(0.5 * (1.0 - tanh(phi_float / delta_GL)))

            elif zeta_pred == True:
                H_k_plus_1 = thick - dt * (
                    q
                    + (ux_s * zs.dx(0) + uy_s * zs.dx(1))
                    - (ux_b * zb.dx(0) + uy_b * zb.dx(1))
                    - a_s + a_b
                )
                phi_pred = bed + (rhoi/rhow) * H_k_plus_1
                zeta.interpolate(0.5 * (1.0 - tanh(phi_pred / delta_GL)))

            grounded = 1.0 - zeta
            grounded_prediction.interpolate(grounded)

            m = Constant(3.0)
            C = beta2 * sqrt(dot(uvec, uvec) + Constant(1.0e-10)**2) ** (1.0/m - 1.0)
            F += grounded * C * dot(uvec, vvect) * ds_b

            if theta_out != 0 and FSSA_keyword == "full":
                
                # First line (excluding first term)
                F += theta * rhoi * g * dt * ((1 - zeta * rhoi/rhow) * (ux_s * zs.dx(0) + uy_s * zs.dx(1) - ux_b * zb.dx(0) - uy_b * zb.dx(1) + q - a_s)- (1 + zeta * rhoi/rhow) * a_b) * (v1.dx(0) + v2.dx(1)) * dx
                
                # Second line
                F += theta * rhoi * g * dt * n[2] * zs * ((1 - zeta * (rhoi/rhow)) * (ux_s * zs.dx(0) + uy_s * zs.dx(1) - ux_b * zb.dx(0) - uy_b * zb.dx(1) + q) + zeta * (rhoi/rhow) * (a_s - a_b) - a_b) * (v1.dx(0) + v2.dx(1)) * ds_t
                
                # Third line
                F += theta * rhoi * g * dt * n[2] * zs * (- zeta * (rhoi/rhow) * ((ux_s * zs.dx(0) + uy_s * zs.dx(1)) - (ux_b * zb.dx(0) + uy_b * zb.dx(1)) + q) + zeta * (rhoi/rhow) * (a_s - a_b) - a_b) * (v1.dx(0) + v2.dx(1)) * ds_b

                # Fourth line
                F -= theta * dt * rhoi * g * a_s * n[2] * (zs.dx(0) * v1 + zs.dx(1) * v2) * ds_t 
                F += theta * zeta * dt * rhoi * g * a_b * n[2] * (zs.dx(0) * v1 + zs.dx(1) * v2)* ds_b
                
                # The constraints:
                F += dot(u_s - uvec, eta) * ds_t
                F += dot(u_b - uvec, xi) * ds_b
                F += r * q * dx - r * thick * (ux.dx(0) + uy.dx(1)) * dx

            if theta_out == 0:
                J = derivative(F, uvec, du)
                problem = NonlinearVariationalProblem(F, uvec, bcs=bcs_u, J=J)
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
                uvec_out.assign(uvec)
                w.sub(0).assign(uvec_out)
            else:
                uvec_out.assign(w.sub(0))

            u_prev.assign(uvec_out)

            print("Solving thickness evolution now...")

            ubar = Function(VVbar, name="u_bar")
            ubar.project(uvec_out)  # testing this without the BCs. Older version:    ubar.project(uvec_out, bcs=bcs_ubar)
            ux_bar, uy_bar = split(ubar)

            vel = as_vector([ux_bar, uy_bar])
            vnorm = sqrt(dot(vel, vel) + 1e-10)
            h = CellDiameter(mesh3D)
            mu_art = 0.1 * h * vnorm

            F = (
                thick_new * phi * dx \
                - thick * phi * dx \
                + dt * (ux_bar * thick_new).dx(0) * phi * dx
                + dt * (uy_bar * thick_new).dx(1) * phi * dx
                - dt * (a_s - a_b) * phi * dx
                # Artifical viscosity
                + dt * mu_art * dot(grad(thick_new), grad(phi)) * dx
            )

            old_thick = Function(Vbar)
            old_thick.assign(thick)

            solve(lhs(F) == rhs(F), H)
            thick.assign(H)
            thick.dat.data[:] = np.maximum(thick.dat.data, 10.0)

            sigma = (SpatialCoordinate(mesh3D)[2] - zb) / old_thick

            zb_float = -rhoi / rhow * thick
            zb.interpolate(max_value(bed, zb_float))
            zs.interpolate(zb + thick)

            mesh3D.coordinates.interpolate(as_vector([xref, yref, zb + sigma * thick]))
            print("Finished solving thickness evolution...")
            

            if restart_from is None:
                print("Year: ", (i+1)*dt)
            else:
                print("Year: ", (t_restart + (i - start_step + 1) * dt))

            if restart_from is None:
                t = (i + 1) * dt
            else:
                t = t_restart + (i - start_step + 1) * dt

            if abs(t % 5) < 1.0e-10 or abs((t % 5) - 5) < 1.0e-10:
                ux_out, uy_out = split(uvec_out)
                ux_s, uy_s = split(u_s)
                ux_b, uy_b = split(u_b)
                uout.interpolate(as_vector([ux_out, uy_out, 0.0]))
                usout.interpolate(as_vector([ux_s, uy_s, 0.0]))
                ubout.interpolate(as_vector([ux_b, uy_b, 0.0]))
                zeta_out.interpolate(zeta_)
                outfile.write(uout, usout, ubout, thick, zs, zb, bed, zeta_out, time=t)
                restart_dir = f"Simulations/MISMIP_output_theta{theta_out:g}_dt{dt:g}_GL_pred{zeta_pred}"
                os.makedirs(restart_dir, exist_ok=True)
                restart_file = f"{restart_dir}/restart_t{t:g}.h5"
                save_restart(restart_file, t, i+1, dt, theta_out)
