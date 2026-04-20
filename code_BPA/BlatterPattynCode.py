from firedrake import *
from netgen.occ import *
import time

'''
Lx = 10000.0
Ly = 10000.0
nz = 10

base = PeriodicRectangleMesh(100, 100, Lx, Ly)
'''

base = UnitSquareMesh(100, 100)
base.coordinates.dat.data[:] *= 10000.0

nz = 10
mesh = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)
x, y, sigma = SpatialCoordinate(mesh)

Xref = Function(mesh.coordinates.function_space(), name="Xref")
Xref.interpolate(SpatialCoordinate(mesh))
xref, yref, sigmaref = split(Xref)

# Horizontal and vertical factors
horiz = FiniteElement("CG", triangle, 1)
vert  = FiniteElement("CG", interval, 1)

# Scalar tensor-product element: CG1 (horizontal) x CG1 (vertical)
scalar_elt = TensorProductElement(horiz, vert)
V = FunctionSpace(mesh, scalar_elt)

vector_elt = VectorElement(scalar_elt, dim=2)
VV = FunctionSpace(mesh, vector_elt)

uvec=Function(VV)
(ux,uy)=split(uvec)

uvect=TrialFunction(VV)
(u1,u2)=split(uvect)

vvect=TestFunction(VV)
(v1,v2)=split(vvect)

# Vector-valued version of the same tensor-product space
vector3_elt = VectorElement(scalar_elt, dim=3)
VV3 = FunctionSpace(mesh, vector3_elt)
uout = Function(VV3, name="uout")

# vertically constant scalar space
Vbar = FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)

# vertically constant 2-component vector space
VVbar = VectorFunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0, dim=2)

phi = TestFunction(Vbar)
thick_new = TrialFunction(Vbar)
H = Function(Vbar)

u_prev = Function(VV)
u_prev_ts = Function(VV)

rhoi = Constant(9.1380e-19)
g = Constant(9.7692e15)

yearinsec = 365.25 * 24 * 60 * 60
A = Constant(1e-26 * yearinsec * 1.0e18)
Amp = Constant(500.0)
x0 = Constant(5000.0)
y0 = Constant(5000.0)
sigma_R = Constant(1000.0)
H0 = Constant(100)
thick = Function(Vbar).interpolate(1000.0 + Amp * exp(-((x - x0)**2 + (y - y0)**2) / sigma_R**2))

mesh.coordinates.interpolate(as_vector([xref, yref, sigmaref * thick]))
eps = Constant(1e-10)

def viscosity(ux, uy, n=1):
    '''
    Double check this against the derivation
    '''
    eps_e2 = (ux.dx(0)**2 + uy.dx(1)**2 + ux.dx(0) * uy.dx(1) \
              + 0.25 * (ux.dx(1) + uy.dx(0))**2 + 0.25 * ux.dx(2)**2
              + 0.25 * uy.dx(2)**2)

    mu = 0.5 * A**(-1.0 / n) * (eps_e2 + eps**2)**((1.0 - n) / (2.0 * n))
    return mu

mu = 1
ns = np.linspace(1, 3, 11)

dt= 0.01                           # Time-step size
theta = Constant(0.0)              # TSS activated: theta=1, TSS deactivated: theta=0
T = 1                              # Simulation length 
num_TS = int(T / dt)

outfile = VTKFile("BPA_output.pvd")
VTKFile("mesh.pvd").write(mesh)

for i in range(num_TS):
    for j, n in enumerate(ns):
        print("Solving with n = ", n)
        change=100
        tol=1e-3
        maxiter=200
        iter_sim=0

        while change>tol and iter_sim<maxiter:
            iter_sim=iter_sim+1

            mu = viscosity(ux, uy, n)

            a = (4 * mu * u1.dx(0) + 2 * mu * u2.dx(1)) * v1.dx(0) * dx \
                + (mu * u1.dx(1) + mu * u2.dx(0)) * v1.dx(1) * dx \
                + mu * u1.dx(2) * v1.dx(2) * dx
            
            a += (4 * mu * u2.dx(1) + 2 * mu * u1.dx(0)) * v2.dx(1) * dx \
                + (mu * u2.dx(0) + mu * u1.dx(1)) * v2.dx(0) * dx \
                + mu * u2.dx(2) * v2.dx(2) * dx

            L = rhoi * g * thick * v1.dx(0) * dx \
            + rhoi * g * thick * v2.dx(1) * dx

            uvecold=uvec.copy(deepcopy=True)
            (uxold,uyold)=split(uvecold)
            print("Solving momentum")
            solve(a == L, uvec)

            du = Function(VV)
            du.assign(uvec)
            du -= uvecold
            change = norm(du) / max(norm(uvec), 1.0e-12)

            u_prev.assign(uvec)
            print("change:", change)

    print("Solving thickness evolution now...")

    # Here, I need to vertically average the velocity...
    ubar = Function(VVbar, name="u_bar")
    ubar.project(uvec)
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
    mesh.coordinates.interpolate(as_vector([xref, yref, sigmaref * thick]))
    print("Finished solving thickness evolution...")
    print("Year: ", (i+1)*dt)

    t = (i + 1) * dt
    uout.interpolate(as_vector([ux, uy, 0.0]))
    outfile.write(uout, thick, time=t)
