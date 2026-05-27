from firedrake import *
import numpy as np
from config import *
from domain import *
from fields import *
from physics import *
from geometry import *
from spaces import *
from bcs import *
from io import *

for dt in dts:
    for theta_out in theta_outs:

        reset_state()
        
        print("=" * 80)
        print(f"Starting run with dt={dt:g}, theta_out={theta_out:g}")
        print("=" * 80)

        theta = Constant(theta_out)
        num_TS = int(T / dt)

        outfile = VTKFile(f"Simulations/MISMIP_output_theta{theta_out:g}_dt{dt:g}.pvd")

        for i in range(num_TS):
            for j, n in enumerate(ns):
                print("Solving with n = ", n)

                if theta_out == 0:
                    uvec = Function(VV)
                    uvec.interpolate(w.sub(0))

                    ux, uy = split(uvec)

                    du = TrialFunction(VV)
                    vvect = TestFunction(VV)
                    v1, v2 = split(vvect)

                else:
                    uvec, u_s, q = split(w)
                    ux, uy = split(uvec)
                    ux_s, uy_s = split(u_s)

                    dw = TrialFunction(W)
                    vvect, eta, r = TestFunctions(W)
                    v1, v2 = split(vvect)

                mu = viscosity(ux, uy, n)

                surf = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)

                F = (4 * mu * ux.dx(0) + 2 * mu * uy.dx(1)) * v1.dx(0) * dx \
                                + (mu * ux.dx(1) + mu * uy.dx(0)) * v1.dx(1) * dx \
                                + mu * ux.dx(2) * v1.dx(2) * dx
                            
                F += (4 * mu * uy.dx(1) + 2 * mu * ux.dx(0)) * v2.dx(1) * dx \
                                + (mu * uy.dx(0) + mu * ux.dx(1)) * v2.dx(0) * dx \
                                + mu * uy.dx(2) * v2.dx(2) * dx

                #delta_zb = Constant(1.0)
                grounded = 1.0 - tanh((zb - bed) / delta_zb)
                zeta = Constant(1.0) - grounded

                F += grounded * beta2 * dot(uvec, vvect) * ds_b

                F -= rhoi * g * zs * (v1.dx(0) + v2.dx(1)) * dx
                
                if theta_out != 0 and FSSA_keyword == "full":


                    F += theta * rhoi * g * dt * (ux_s * zs.dx(0) + uy_s * zs.dx(1)) * (v1.dx(0) + v2.dx(1)) * dx
                    F += theta * rhoi * g * dt * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) * q * (v1.dx(0) + v2.dx(1)) * dx
                    F += 0.5 * theta * rhoi * g * dt * (ux_s * (zs * zs).dx(0) + uy_s * (zs * zs).dx(1)) * (v1.dx(0) + v2.dx(1)) * surf * ds_t
                    F += theta * rhoi * g * dt * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) * zs * (v1.dx(0) + v2.dx(1)) * surf * q * ds_t
                
                    # The constraints:
                    F += dot(u_s - uvec, eta) * ds_t
                    F += r * q * dx - r * thick * (ux.dx(0) + uy.dx(1)) * dx

                    # The accumulation terms:
                    n_z = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)
                    F -= theta * a_s * n_z * rhoi * g * dt * (zs.dx(0) * v1 + zs.dx(1) * v2) * ds_t
                    F -= theta * rhoi * g * dt * a_s * (v1.dx(0) + v2.dx(1)) * dx

                # The stabilisation terms (the version where we assume that test function gradients are depth independent)
                if theta_out != 0 and FSSA_keyword == "approx":
                    # This terms needs to be updated
                    F += theta * rhoi * g * dt * thick * (ux * zs.dx(0) + uy * zs.dx(1)) * (v1.dx(0) + v2.dx(1)) * surf * ds_t
                    F += theta * rhoi * g * dt * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) * thick * (ux.dx(0) + uy.dx(1)) * (v1.dx(0) + v2.dx(1)) * dx
                    F += 0.5 * theta * rhoi * g * dt * (ux * (zs * zs).dx(0) + uy * (zs * zs).dx(1)) * (v1.dx(0) + v2.dx(1)) * surf * ds_t
                    F += theta * rhoi * g * dt * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) * zs * (v1.dx(0) + v2.dx(1)) * surf * thick * (ux.dx(0) + uy.dx(1)) * ds_t

                    # The accumulation terms:
                    n_z = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)
                    F -= theta * a_s * n_z * rhoi * g * dt * (zs.dx(0) * v1 + zs.dx(1) * v2) * ds_t
                    F -= theta * rhoi * g * dt * a_s * (v1.dx(0) + v2.dx(1)) * dx

                z = SpatialCoordinate(mesh3D)[2]
                sea_level = Constant(0.0)

                p_ocean = rhow * g * max_value(sea_level - z, 0.0)

                # To do: Get the ocean pressure properly set up. The current version is close, but not strictly accurate
                #F += p_ocean * v1 * ds_v(2)

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
            ubar.project(uvec_out, bcs=bcs_ubar)
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

            solve(lhs(F) == rhs(F), H)
            thick.assign(H)
            thick.dat.data[:] = np.maximum(thick.dat.data, 10.0)

            zb_float = -rhoi / rhow * thick
            zb.interpolate(max_value(bed, zb_float))
            zs.interpolate(zb + thick)

            mesh3D.coordinates.interpolate(as_vector([xref, yref, zb + sigmaref * thick]))
            print("Finished solving thickness evolution...")
            print("Year: ", (i+1)*dt)

            t = (i + 1) * dt

            if abs(t % 50) < 1.0e-10 or abs((t % 50) - 50) < 1.0e-10:
                ux_out, uy_out = split(uvec_out)
                uout.interpolate(as_vector([ux_out, uy_out, 0.0]))
                grounded_out.interpolate(1.0 - tanh((zb - bed) / delta_zb))
                outfile.write(uout, thick, zs, zb, grounded_out, time=t)
                restart_file = f"Simulations/restart_theta{theta_out:g}_dt{dt:g}_t{t:g}.h5"
                save_restart(restart_file, t, i+1, dt, theta_out)
