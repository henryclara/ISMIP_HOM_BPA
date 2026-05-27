from firedrake import *
import numpy as np
from config import *
from domain import *
# -----------------------------
# Elements
# -----------------------------

# Horizontal and vertical elements
horiz = FiniteElement("CG", triangle, 1)
vert  = FiniteElement("CG", interval, 1)

scalar_elt = TensorProductElement(horiz, vert)

vector_elt = VectorElement(scalar_elt, dim=2)

vector3_elt = VectorElement(scalar_elt, dim=3)

# -----------------------------
# Function spaces
# -----------------------------

V = FunctionSpace(mesh3D, scalar_elt)
VV = FunctionSpace(mesh3D, vector_elt)

VV3 = FunctionSpace(mesh3D, vector3_elt)
uout = Function(VV3, name="uout")

Vbar = FunctionSpace(mesh3D, "CG", 1, vfamily="R", vdegree=0)
VVbar = VectorFunctionSpace(mesh3D, "CG", 1, vfamily="R", vdegree=0, dim=2)

H = Function(Vbar)
u_prev = Function(VV)

uvec_out = Function(VV, name="uvec")

# -----------------------------
# Mixed spaces
# -----------------------------

W = VV * VVbar * Vbar
w = Function(W)

uvec, u_s, q = split(w)
ux, uy = split(uvec)
ux_s, uy_s = split(u_s)

# -----------------------------
# Test functions
# -----------------------------

vvect, eta, r = TestFunctions(W)
v1, v2 = split(vvect)

phi = TestFunction(Vbar)

# -----------------------------
# Trial functions
# -----------------------------

thick_new = TrialFunction(Vbar)
