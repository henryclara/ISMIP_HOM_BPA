
from firedrake import *
from netgen.occ import *
import numpy as np
import time

Lx = 80000.0
Ly = 80000.0
nz = 10

base = PeriodicRectangleMesh(50, 50, Lx, Ly)

nz = 10
mesh = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)
x, y, sigma = SpatialCoordinate(mesh)

Xref = Function(mesh.coordinates.function_space(), name="Xref")
Xref.interpolate(SpatialCoordinate(mesh))
xref, yref, sigmaref = split(Xref)

# Horizontal and vertical elements
horiz = FiniteElement("CG", triangle, 1)
vert  = FiniteElement("CG", interval, 1)

# Scalar tensor-product element: CG1 (horizontal) x CG1 (vertical)
scalar_elt = TensorProductElement(horiz, vert)
V = FunctionSpace(mesh, scalar_elt)

vector_elt = VectorElement(scalar_elt, dim=2)
VV = FunctionSpace(mesh, vector_elt)

# Vector-valued version of the same tensor-product space
vector3_elt = VectorElement(scalar_elt, dim=3)
VV3 = FunctionSpace(mesh, vector3_elt)
uout = Function(VV3, name="uout")

Vbar = FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)
VVbar = VectorFunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0, dim=2)

#W = VV * Vbar
#w = Function(W)

W = VV * VVbar * Vbar
w = Function(W)

uvec, u_s, q = split(w)
ux, uy = split(uvec)
ux_s, uy_s = split(u_s)

vvect, eta, r = TestFunctions(W)
v1, v2 = split(vvect)

#dw = TrialFunction(W)
#vvect, r = TestFunctions(W)
#v1, v2 = split(vvect)

uvec_out = Function(VV, name="uvec")

phi = TestFunction(Vbar)
thick_new = TrialFunction(Vbar)
H = Function(Vbar)

u_prev = Function(VV)
u_prev_ts = Function(VV)

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
             + 500.0 * sin(omega * x) * sin(omega * y))

thick = Function(Vbar, name="thick").interpolate(zs - zb)

mesh.coordinates.interpolate(as_vector([xref, yref, zb + sigmaref * thick]))
eps = Constant(1e-6) # Constant(1e-10)

def viscosity(ux, uy, n=1):
    '''
    Double check this against the derivation
    '''
    eps_e2 = (ux.dx(0)**2 + uy.dx(1)**2 + ux.dx(0) * uy.dx(1) \
              + 0.25 * (ux.dx(1) + uy.dx(0))**2 + 0.25 * ux.dx(2)**2 \
              + 0.25 * uy.dx(2)**2)

    mu = 0.5 * A**(-1.0 / n) * (eps_e2 + eps**2)**((1.0 - n) / (2.0 * n))
    return mu

mu = 1
ns = [3] #np.linspace(1, 3, 11)

# Basal friction field
beta2 = Function(Vbar, name="beta2")
beta2.interpolate(1000.0 * (1.0 + sin(2.0*np.pi*x/Lx) * sin(2.0*np.pi*y/Lx)))

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
        zb.interpolate(zs - 1000.0 + 500.0 * sin(omega * x) * sin(omega * y))
        thick.interpolate(zs - zb)
        mesh.coordinates.interpolate(as_vector([xref, yref, zb + sigmaref * thick]))

        u_init = as_vector([10.0 * (1.0 + 0.01 * sin(2.0*pi*x/Lx) * sin(2.0*pi*y/Ly)), \
                            10.0 * (0.01 * sin(2.0*pi*x/Lx) * sin(2.0*pi*y/Ly)),])

        w.sub(0).interpolate(u_init)   # initial guess for u
        w.sub(1).interpolate(u_init)   # initial guess for u_s

        u0 = w.sub(0)
        ux0, uy0 = split(u0)

        w.sub(2).project(thick * (ux0.dx(0) + uy0.dx(1)))

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
                    uvec = Function(VV)
                    uvec.interpolate(w.sub(0))   # use current mixed velocity guess if available

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

                grad_zs_H = as_vector([zs.dx(0), zs.dx(1)])
                surf = 1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)

                #q.project(thick * (ux.dx(0) + uy.dx(1)))

                F = (4 * mu * ux.dx(0) + 2 * mu * uy.dx(1)) * v1.dx(0) * dx \
                                + (mu * ux.dx(1) + mu * uy.dx(0)) * v1.dx(1) * dx \
                                + mu * ux.dx(2) * v1.dx(2) * dx
                            
                F += (4 * mu * uy.dx(1) + 2 * mu * ux.dx(0)) * v2.dx(1) * dx \
                                + (mu * uy.dx(0) + mu * ux.dx(1)) * v2.dx(0) * dx \
                                + mu * uy.dx(2) * v2.dx(2) * dx
                            
                F += beta2 * dot(uvec, vvect) * ds_b
                
                if theta_out != 0:
                    
                    # Second term...
                    F += theta * rhoi * g * dt * (ux_s * zs.dx(0) + uy_s * zs.dx(1)) * (v1.dx(0) + v2.dx(1)) * dx
                    
                    # Third term
                    F += theta * rhoi * g * np.cos(psi) * dt * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) * q * (v1.dx(0) + v2.dx(1)) * dx
                
                    # Fourth term
                    F += 0.5 * theta * g * np.cos(psi) * dt * (ux * (zs * zs).dx(0) + uy * (zs * zs).dx(1)) * (v1.dx(0) + v2.dx(1)) * (1 / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)) * ds_t

                    # Fifth term
                    F += theta * rhoi * g * np.cos(psi) * dt * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) * zs * ((v1.dx(0) + v2.dx(1)) / sqrt(1 + zs.dx(0)**2 + zs.dx(1)**2)) * q * ds_t
                
                    # The constraints:
                    F += dot(u_s - uvec, eta) * ds_t
                    F += r * q * dx - r * thick * (ux.dx(0) + uy.dx(1)) * dx

                # The stabilisation terms (the version where we assume that test function gradients are depth independent)
                '''
                if theta_out != 0:
                    F += theta * rhoi * g * np.cos(psi) * dt * thick * (ux * zs.dx(0) + uy * zs.dx(1)) \
                                    * (v1.dx(0) + v2.dx(1)) * surf * ds_t

                    F += theta * rhoi * g * np.cos(psi) * dt * (((1 - zeta) * rhoi - rhow)/(rhoi - rhow)) \
                                    * q * (v1.dx(0) + v2.dx(1)) * dx

                    div_h = ux.dx(0) + uy.dx(1)

                    F += r * q * dx - r * thick * div_h * dx

                '''

                F -= rhoi * g * np.cos(psi) * zs * (v1.dx(0) + v2.dx(1)) * dx \
                            - rhoi * g * np.sin(psi) * v1 * dx
                    
                    # Ignore accumulation for now.
                    #- theta * rhoi * g * dt * (a_s - a_b) * zb.dx(0) * v1 * dx \
                    #- theta * rhoi * g * dt * (a_s - a_b) * zb.dx(1) * v2 * dx \
                    #+ theta * rhoi * g * dt * thick * (a_s - a_b) * v1.dx(0) * dx \
                    #+ theta * rhoi * g * dt * thick * (a_s - a_b) * v2.dx(1) * dx

                #J = derivative(F, uvec, du)
                #problem = NonlinearVariationalProblem(F, uvec, J=J)

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

            # This needs improvement. We need to solve a weak form problem here.
            ubar.project(uvec_out)
            ux_bar, uy_bar = split(ubar)

            vel = as_vector([ux_bar, uy_bar])
            vnorm = sqrt(dot(vel, vel) + 1e-10)
            h = CellDiameter(mesh)
            mu_art = 0.1 * h * vnorm

            F = (
                thick_new * phi * dx \
                - thick * phi * dx \
                + dt * (ux_bar * thick_new).dx(0) * phi * dx
                + dt * (uy_bar * thick_new).dx(1) * phi * dx
                # Artifical viscosity
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
