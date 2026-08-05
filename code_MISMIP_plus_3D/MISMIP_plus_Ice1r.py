
from firedrake import *
import numpy as np
import os
from config import *
from datetime import datetime

for dt in dts:
    for theta_out in theta_outs:

        start_step = 0
        restart_from = f"Simulations/MISMIP_output_theta1_dt10_GL_predFalse/restart_t10000.h5" # Change to whatever is correct
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
        zeta_im = Function(Q1, name="zeta")
        zeta_out = Function(Q1, name="zeta_out")
        zeta_predicted = Function(Q1, name="zeta_predicted")

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
        outfile = VTKFile(f"Simulations/MISMIP_Ice1r_theta{theta_out:g}_dt{dt:g}_GL_pred{zeta_pred}.pvd")

        if theta_out == 0:
            uvec = Function(VV)
            uvec.assign(w.sub(0))

            ux, uy = split(uvec)

            du = TrialFunction(VV)
            vvect = TestFunction(VV)
            v1, v2 = split(vvect)

        else:
            uvec, q = split(w)

            ux, uy = split(uvec)
            q1, q2 = split(q)

            dw = TrialFunction(W)

            vvect, r = TestFunctions(W)
            v1, v2 = split(vvect)

        for i in range(start_step, num_TS):
            print("Solving momentum")

            mu = viscosity(ux, uy, 3.0)

            cavity_thickness = max_value(zb - bed, 0.0)

            a_b_Ice1r = (Constant(0.2) * tanh(cavity_thickness / Constant(75.0)) \
                                    * max_value(-Constant(100.0) - zb, Constant(0.0)))

            surf = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)

            F = (4 * mu * ux.dx(0) + 2 * mu * uy.dx(1)) * v1.dx(0) * dx \
                            + (mu * ux.dx(1) + mu * uy.dx(0)) * v1.dx(1) * dx \
                            + mu * ux.dx(2) * v1.dx(2) * dx
                        
            F += (4 * mu * uy.dx(1) + 2 * mu * ux.dx(0)) * v2.dx(1) * dx \
                            + (mu * uy.dx(0) + mu * ux.dx(1)) * v2.dx(0) * dx \
                            + mu * uy.dx(2) * v2.dx(2) * dx

            phi_float = bed + (rhoi/rhow) * thick
            delta_GL = Constant(0.01)
            grounded_prediction = Function(Q1, name="grounded_prediction")

            n = FacetNormal(mesh3D)

            F -= rhoi * g * zs * (v1.dx(0) + v2.dx(1)) * dx

            eps_H = Constant(1.0e-10)

            p_W = rhow * g * max_value(0.0, thick - zs)
            p_I = rhoi * g * max_value(thick, eps_H)

            zeta.interpolate(min_value(1.0, max_value(0.0, p_W / p_I)))

            if zeta_pred == True:
                H_k_plus_1 = thick - dt * (q1.dx(0) + q2.dx(1) - a_s + a_b_Ice1r)
                phi_pred = bed + (rhoi/rhow) * H_k_plus_1
                zeta_im.interpolate(0.5 * (1.0 - tanh(phi_pred / delta_GL)))
                
            zeta_predicted.assign(zeta_im)

            m = Constant(3.0)

            C = beta2 * sqrt(dot(uvec, uvec) + Constant(1.0e-10)**2) ** (1.0/m - 1.0)

            if zeta_pred == True:
                F += (1 - zeta_im) * C * dot(uvec, vvect) * ds_b

            elif zeta_pred == False:
                F += (1 - zeta) * C * dot(uvec, vvect) * ds_b

            gamma = rhoi * g * (1 - (rhoi/rhow))
            gdash = rhoi * g

            if theta_out != 0 and FSSA_keyword == "full":
                # First line (excluding first term)
                F -= ((1 - zeta) * gdash + zeta * gamma) * dt * (-q1.dx(0) - q2.dx(1) + a_s - zeta * a_b_Ice1r) * (v1.dx(0) + v2.dx(1)) * dx

                # Second line
                F -= ((1 - zeta) * gdash + zeta * gamma) * dt * n[2] * (-q1.dx(0) - q2.dx(1) + a_s - zeta * a_b_Ice1r) * (v1 * zs.dx(0) + v2 * zs.dx(1)) * ds_t

                # Third line
                F -= ((1 - zeta) * gdash + zeta * gamma) * dt * n[2] * (-q1.dx(0) - q2.dx(1) + a_s - zeta * a_b_Ice1r) * (v1 * zs.dx(0) + v2 * zb.dx(1)) * ds_b
                
                F += dot(q - thick * uvec, r) * dx

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
            ubar.project(uvec_out)
            ux_bar, uy_bar = split(ubar)

            vel = as_vector([ux_bar, uy_bar])
            vnorm = sqrt(dot(vel, vel) + 1e-10)
            h = CellDiameter(mesh3D)
            mu_art = 0.1 * h * vnorm

            z_ref = Function(Q1, name="z_ref")
            z_ref.interpolate(SpatialCoordinate(mesh3D)[2])

            sigma_ref = Function(Q1, name="sigma_ref")
            sigma_ref.interpolate((z_ref - zb) / thick)

            F = (
                thick_new * phi * dx \
                - thick * phi * dx \
                + dt * (ux_bar * thick_new).dx(0) * phi * dx
                + dt * (uy_bar * thick_new).dx(1) * phi * dx
                - dt * (a_s - a_b_Ice1r) * phi * dx
                # Artifical viscosity
                + dt * mu_art * dot(grad(thick_new), grad(phi)) * dx
            )

            old_thick = Function(Vbar)
            old_thick.assign(thick)

            solve(lhs(F) == rhs(F), H)
            thick.assign(H)
            thick.dat.data[:] = np.maximum(thick.dat.data, 10.0)

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
                ux_out, uy_out = split(uvec_out)
                uout.interpolate(as_vector([ux_out, uy_out, 0.0]))
                zeta_out.interpolate(zeta)
                outfile.write(uout, thick, zs, zb, bed, zeta_out, zeta_predicted, time=t)
                restart_dir = f"Simulations/MISMIP_Ice1r_theta{theta_out:g}_dt{dt:g}_GL_pred{zeta_pred}"
                os.makedirs(restart_dir, exist_ok=True)
                restart_file = f"{restart_dir}/restart_t{t:g}.h5"
                save_restart(restart_file, t, i+1, dt, theta_out)

        simulation_end = datetime.now()
        print(f"Simulation time: {simulation_end - simulation_start}")
