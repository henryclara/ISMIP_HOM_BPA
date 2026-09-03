import os

# Must be set BEFORE importing Firedrake.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from pathlib import Path
from datetime import datetime

from firedrake import *
from config import *

if restart_from is not None:
    os.environ["RESTART_MESH_FILE"] = str(restart_from)
else:
    os.environ.pop("RESTART_MESH_FILE", None)

from fields import *
from physics import *
from geometry import *
from spaces import *
from bcs import *
from io_local import *
import ufl

Q1 = FunctionSpace(mesh3D, "CG", 1)

if restart_from is None:
    mesh3D.coordinates.interpolate(
        as_vector([xref, yref, zb + zref * thick])
    )

restart_coordinates = Function(
    mesh3D.coordinates.function_space()
)
restart_coordinates.assign(mesh3D.coordinates)

for dt in dts:
    for theta_out in theta_outs:

        simulation_start = datetime.now()

        mesh3D.coordinates.assign(restart_coordinates)

        if restart_from is not None:
            (t_restart,saved_step,dt_restart,theta_restart) = load_restart(str(restart_from))
        else:
            t_restart = 0.0
            saved_step = 0
            dt_restart = dt
            theta_restart = theta_out

        u_prev.assign(uvec_out)
        H.assign(thick)

        thick_2D.dat.data[:] = thick.dat.data_ro[:]
        zb_2D.dat.data[:] = zb.dat.data_ro[:]
        bed_2D.dat.data[:] = bed.dat.data_ro[:]

        H_2D.assign(thick_2D)

        zeta_out = Function(Q1, name="zeta_out")
        phi_GL_out = Function(Q1, name="flotation_function")

        if restart_from is not None:
            print(
                f"Restarting dt={dt:g}, theta={theta_out:g} "
                f"from t={t_restart:g}"
            )
        
        print("=" * 80)
        print(f"Starting run with dt={dt:g}, theta_out={theta_out:g}")
        print("=" * 80)

        theta = Constant(theta_out)

        # T is the final physical time, not the number of steps.
        remaining_time = T - t_restart
        num_steps = int(round(remaining_time / dt))

        if num_steps <= 0:
            raise ValueError(
                f"No time steps requested: "
                f"T={T:g}, restart time={t_restart:g}, dt={dt:g}"
            )

        # Separate directory for each dt/theta combination.
        run_dir = Path(f"Simulations/{exp_name}_theta{theta_out:g}_dt{dt:g}_res{int(coarse_res)}_{int(refined_res)}_nz{nz}_GJP_2D_thickness")
        run_dir.mkdir(parents=True, exist_ok=True)

        theta = Constant(theta_out)
        num_TS = int(T / dt)
        outfile = VTKFile(str(run_dir / f"{exp_name}.pvd"))

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

        # 100 and 200 model years after the restart.
        runtime_milestones = [
            t_restart + 100.0,
            t_restart + 200.0,
        ]

        logged_runtime_milestones = set()

        for local_step in range(num_steps):
            step_number = local_step + 1

            print("Solving momentum")

            mu = viscosity(ux, uy, 3.0)

            cavity_thickness = max_value(zb - bed, 0.0)

            if exp_name == "Ice1rr" or (exp_name == "Ice1ra" and t<10100):
                a_b_Ice1r = (Constant(0.2) * tanh(cavity_thickness / Constant(75.0)) \
                                        * max_value(-Constant(100.0) - zb, Constant(0.0)))
            else:
                a_b_Ice1r = Constant(0.0)

            surf = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)

            F = (4 * mu * ux.dx(0) + 2 * mu * uy.dx(1)) * v1.dx(0) * dx \
                            + (mu * ux.dx(1) + mu * uy.dx(0)) * v1.dx(1) * dx \
                            + mu * ux.dx(2) * v1.dx(2) * dx
                        
            F += (4 * mu * uy.dx(1) + 2 * mu * ux.dx(0)) * v2.dx(1) * dx \
                            + (mu * uy.dx(0) + mu * ux.dx(1)) * v2.dx(0) * dx \
                            + mu * uy.dx(2) * v2.dx(2) * dx

            n = FacetNormal(mesh3D)

            F -= rhoi * g * zs * (v1.dx(0) + v2.dx(1)) * dx

            z = SpatialCoordinate(mesh3D)[2]
            p_o = conditional(z < 0.0, -rhow * g * z, 0.0)
            calving_traction = (rhoi * g * (zs - z) - p_o)

            F -= calving_traction * (n[0] * v1 + n[1] * v2) * ds(2, domain=mesh3D)

            # Hydrostatic flotation function
            phi_GL = bed + (rhoi/rhow) * thick

            # Discontinuous grounding/floating mask
            # zeta = 0 grounded, 1 floating
            '''
            if zeta_pred:
                if theta_out == 0:
                    div_flux = (thick * ux).dx(0) + (thick * uy).dx(1)
                else:
                    div_flux = q1.dx(0) + q2.dx(1)

                H_pred = thick - dt * div_flux + dt * (a_s - a_b_Ice1r)
                phi_GL_pred = bed + (rhoi/rhow) * H_pred

                zeta_fric = 0.5 * (1.0 - tanh(phi_GL_pred / delta_GL))
            else:
                zeta_fric = conditional(phi_GL >= 0.0, 0.0, 1.0)
            '''
            if zeta_pred:
                #H_pred = thick - dt * (q1.dx(0) + q2.dx(1)) + dt * (a_s - a_b_Ice1r)
                phi_GL_pred = bed + (rhoi/rhow) * (thick - dt * (q1.dx(0) + q2.dx(1)) + dt * (a_s - a_b_Ice1r))
                zeta = conditional(bed + (rhoi/rhow) * (thick - dt * (q1.dx(0) + q2.dx(1)) + dt * (a_s - a_b_Ice1r)) >= 0.0, 0.0, 1.0)

            else:
                zeta = conditional(phi_GL >= 0.0, 0.0, 1.0)

            #zeta = conditional(phi_GL >= 0.0, 0.0, 1.0)

            m = Constant(3.0)

            C = beta2 * sqrt(dot(uvec, uvec) + Constant(1.0e-10)**2) ** (1.0/m - 1.0)

            gl_quad_degree = 3 # 1 and 5km meshes, integration points: 3 and 79 (Seroussi2014)
            F += ((1.0 - zeta) * C * dot(uvec, vvect) * ds_b(degree=gl_quad_degree))

            gamma = rhoi * g * (1 - (rhoi/rhow))

            if theta_out != 0 and FSSA_keyword == "full":
                # First line (excluding first term)
                F -= ((1 - zeta) * rhoi * g  + zeta * gamma) * dt * (-q1.dx(0) - q2.dx(1) + a_s - zeta * a_b_Ice1r) * (v1.dx(0) + v2.dx(1)) * dx

                # Second line
                F -= ((1 - zeta) * rhoi * g + zeta * gamma) * dt * n[2] * (-q1.dx(0) - q2.dx(1) + a_s - zeta * a_b_Ice1r) * (v1 * zs.dx(0) + v2 * zs.dx(1)) * ds_t

                # Third line
                F -= ((1 - zeta) * rhoi * g + zeta * gamma) * dt * n[2] * (-q1.dx(0) - q2.dx(1) + a_s - zeta * a_b_Ice1r) * (v1 * zb.dx(0) + v2 * zb.dx(1)) * ds_b
                
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

            try:
                solver.solve()
            except Exception as exc:
                print("\n" + "!" * 80)
                print(f"SIMULATION FAILED")
                print(exc)
                print("!" * 80 + "\n")
                break

            nonlinear_iterations = solver.snes.getIterationNumber()
            print(
                f"Time step {step_number}: "
                f"t={t_restart + step_number * dt:g}, "
                f"Newton iterations={nonlinear_iterations}"
            )

            if theta_out == 0:
                uvec_out.assign(uvec)
                w.sub(0).assign(uvec_out)
            else:
                uvec_out.assign(w.sub(0))

            u_prev.assign(uvec_out)

            print("Solving thickness evolution now...")

            ubar = Function(VVbar, name="u_bar")
            ubar.project(uvec_out)

            ubar_2D.dat.data[:] = ubar.dat.data_ro[:]
            thick_2D.dat.data[:] = thick.dat.data_ro[:]
            zb_2D.dat.data[:] = zb.dat.data_ro[:]
            bed_2D.dat.data[:] = bed.dat.data_ro[:]

            z_ref = Function(Q1, name="z_ref")
            z_ref.interpolate(SpatialCoordinate(mesh3D)[2])

            sigma_ref = Function(Q1, name="sigma_ref")
            sigma_ref.interpolate((z_ref - zb) / thick)

            cavity_thickness_2D = max_value(zb_2D - bed_2D, Constant(0.0))

            if exp_name == "Ice1rr" or (exp_name == "Ice1ra" and t < 10100):
                a_b_Ice1r_2D = (Constant(0.2) * tanh(cavity_thickness_2D / Constant(75.0))
                    * max_value(-Constant(100.0) - zb_2D, Constant(0.0)))
            else:
                a_b_Ice1r_2D = Constant(0.0)

            if time_stepping == "ex":
                H_adv = thick_2D

            elif time_stepping == "im":
                H_adv = thick_new_2D

            elif time_stepping == "im_mi":
                H_adv = 0.5 * (thick_new_2D + thick_2D)

            else:
                raise ValueError(f"Unknown time stepping option: {time_stepping}")

            n_H = FacetNormal(base)
            h_H = CellDiameter(base)
            h_face = avg(h_H)

            u_n = 0.5 * (abs(dot(ubar_2D("+"), n_H("+"))) + abs(dot(ubar_2D("-"), n_H("-"))))
            gjp = (tau_gjp * h_face**2 * u_n * jump(grad(H_adv), n_H) * jump(grad(phi_2D), n_H) * dS(domain=base))
            dx_H = dx(domain=base)
            ux_bar_2D, uy_bar_2D = split(ubar_2D)

            F_H = (thick_new_2D * phi_2D * dx_H
                - thick_2D * phi_2D * dx_H
                + dt * (ux_bar_2D * H_adv).dx(0) * phi_2D * dx_H
                + dt * (uy_bar_2D * H_adv).dx(1) * phi_2D * dx_H
                - dt * (a_s - a_b_Ice1r_2D) * phi_2D * dx_H
                 + dt * gjp
            )

            solve(lhs(F_H) == rhs(F_H),H_2D)

            thick_2D.assign(H_2D)
            thick_2D.dat.data[:] = np.maximum(thick_2D.dat.data, 10.0)
            thick.dat.data[:] = thick_2D.dat.data_ro[:]
            zb_float = -rhoi / rhow * thick

            zb.interpolate(max_value(bed, zb_float))
            zs.interpolate(zb + thick)
            mesh3D.coordinates.interpolate(as_vector([xref, yref, zb + sigma_ref * thick]))

            print("Finished solving thickness evolution...")

            t = t_restart + step_number * dt
            print("Year:", t)

            for milestone in runtime_milestones:

                if (
                    milestone not in logged_runtime_milestones
                    and abs(t - milestone) < 1.0e-10
                ):

                    runtime_checkpoint_end = datetime.now()
                    runtime_checkpoint = (
                        runtime_checkpoint_end
                        - simulation_start
                    )

                    elapsed_years = (
                        t - t_restart
                    )

                    print(
                        f"Runtime after {elapsed_years:g} model years: "
                        f"{runtime_checkpoint}"
                    )

                    os.makedirs(
                        "Simulations",
                        exist_ok=True,
                    )

                    with open(
                        "Simulations/MISMIP_Ice1r_simulation_times.txt",
                        "a",
                    ) as f:

                        f.write(
                            f"dt={dt:g}, "
                            f"theta_out={theta_out:g}, "
                            f"resolution=500, "
                            f"T_start={t_restart:g}, "
                            f"T_end={t:g}, "
                            f"elapsed_years={elapsed_years:g}, "
                            f"num_steps={step_number}, "
                            f"start={simulation_start}, "
                            f"end={runtime_checkpoint_end}, "
                            f"runtime={runtime_checkpoint}\n"
                        )

                    logged_runtime_milestones.add(
                        milestone
                    )

            if abs(t % output_int) < 1.0e-10 or abs((t % output_int) - output_int) < 1.0e-10:

                ux_out, uy_out = split(uvec_out)
                uout.interpolate(as_vector([ux_out, uy_out, 0.0]))
                phi_GL_out.interpolate(bed + (rhoi/rhow) * thick)

                grounded_out.interpolate(conditional(phi_GL_out > 0.0,1.0,0.0))
                zeta_out.interpolate(zeta)
                outfile.write(uout, thick, zs, zb, bed, zeta_out, grounded_out, phi_GL_out, time=t)
                restart_file = run_dir / f"restart_t{t:g}.h5"

                save_restart(str(restart_file),t,step_number,dt,theta_out)
