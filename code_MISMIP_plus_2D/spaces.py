from firedrake import *
import numpy as np
from config import *
from domain import *
# -----------------------------
# Elements
# -----------------------------

# Horizontal and vertical elements: the 2D slice is now x-z only.
horiz = FiniteElement("CG", interval, 1)
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
usout = Function(VV3, name="usout")
ubout = Function(VV3, name="ubout")

Vbar = FunctionSpace(mesh3D, "CG", 1, vfamily="R", vdegree=0)
VVbar = VectorFunctionSpace(mesh3D, "CG", 1, dim=1)

H = Function(Vbar)
u_prev = Function(V)

uvec_out = Function(V, name="uvec")

# -----------------------------
# Mixed spaces
# -----------------------------

W = V * Vbar * Vbar * Vbar
w = Function(W)

u, u_s, u_b, q_mixed = split(w)

q = Function(Vbar, name="q")

# -----------------------------
# Test functions
# -----------------------------

v, eta, xi, r = TestFunctions(W)

phi = TestFunction(Vbar)

# -----------------------------
# Trial functions
# -----------------------------

thick_new = TrialFunction(Vbar)
