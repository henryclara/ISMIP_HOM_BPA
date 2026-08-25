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

W = VV * VVbar
w = Function(W)

uvec, q = split(w)
ux, uy = split(uvec)
#ux_s, uy_s = split(u_s)
#ux_b, uy_b = split(u_b)

# -----------------------------
# Test functions
# -----------------------------

vvect, r = TestFunctions(W)
v1, v2 = split(vvect)

phi = TestFunction(Vbar)

# -----------------------------
# Trial functions
# -----------------------------

thick_new = TrialFunction(Vbar)

# ============================================================
# 2D horizontal spaces for thickness evolution
# ============================================================

Q_H = FunctionSpace(base, "CG", 1)

V_H = VectorFunctionSpace(
    base,
    "CG",
    1,
    dim=2,
)

# Thickness solution
H_2D = Function(Q_H, name="H_2D")

# Vertically averaged velocity represented on the base mesh
ubar_2D = Function(V_H, name="u_bar_2D")

# Test/trial functions for horizontal thickness equation
phi_2D = TestFunction(Q_H)
thick_new_2D = TrialFunction(Q_H)