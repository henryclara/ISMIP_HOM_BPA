from firedrake import *

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

W = VV * VVbar * Vbar
w = Function(W)

uvec, u_s, q = split(w)
ux, uy = split(uvec)
ux_s, uy_s = split(u_s)

vvect, eta, r = TestFunctions(W)
v1, v2 = split(vvect)

uvec_out = Function(VV, name="uvec")

phi = TestFunction(Vbar)
thick_new = TrialFunction(Vbar)
H = Function(Vbar)

u_prev = Function(VV)
