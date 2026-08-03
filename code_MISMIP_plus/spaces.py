from firedrake import *
from config import *
from domain import *

# -----------------------------
# Function spaces for the 2D setup
# -----------------------------

V = FunctionSpace(mesh3D, "CG", 1)
VV = VectorFunctionSpace(mesh3D, "CG", 1, dim=1)
VVbar = VectorFunctionSpace(mesh3D, "CG", 1, dim=1)
Vbar = FunctionSpace(mesh3D, "CG", 1)

uout = Function(VV, name="uout")

H = Function(Vbar, name="H")
u_prev = Function(VV, name="u_prev")
uvec_out = Function(VV, name="uvec_out")

# -----------------------------
# Scalar and vector test/trial functions
# -----------------------------

phi = TestFunction(Vbar)
thick_new = TrialFunction(Vbar)

uvec = TrialFunction(VV)
du = TrialFunction(VV)
vvect = TestFunction(VV)
v1 = split(vvect)[0]
