from firedrake import *

from spaces import *

bcs_u = [
    DirichletBC(VV.sub(0), 0.0, (1, 3, 4)),
]

bcs_ubar = [
    DirichletBC(VVbar.sub(0), 0.0, (1, 3, 4)),
]
