from firedrake import *
import numpy as np

from spaces import *

# For theta_out == 0
bcs_u = [
    DirichletBC(V, 0.0, (1, 2)),
]

# For theta_out != 0
bcs_w = [
    DirichletBC(W.sub(0), 0.0, (1, 2)),
    DirichletBC(W.sub(1), 0.0, (1, 2)),
    DirichletBC(W.sub(2), 0.0, (1, 2)),
]

bcs_ubar = [
    DirichletBC(Vbar, 0.0, (1, 2))
]
