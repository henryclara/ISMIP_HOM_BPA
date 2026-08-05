from firedrake import *
import numpy as np

from spaces import *

# For theta_out == 0
bcs_u = [
    DirichletBC(VV.sub(0), 0.0, 1),
    DirichletBC(VV.sub(1), 0.0, 3),
    DirichletBC(VV.sub(1), 0.0, 4),
]

# For theta_out != 0
bcs_w = [
    # uvec side impenetrability
    DirichletBC(W.sub(0).sub(0), 0.0, 1),
    DirichletBC(W.sub(0).sub(1), 0.0, 3),
    DirichletBC(W.sub(0).sub(1), 0.0, 4),

    DirichletBC(W.sub(1).sub(0), 0.0, 1),
    DirichletBC(W.sub(1).sub(1), 0.0, 3),
    DirichletBC(W.sub(1).sub(1), 0.0, 4),
]

bcs_ubar = [
    DirichletBC(VVbar.sub(0), 0.0, 1),
    DirichletBC(VVbar.sub(1), 0.0, 3),
    DirichletBC(VVbar.sub(1), 0.0, 4),
]
