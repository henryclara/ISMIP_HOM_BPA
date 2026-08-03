from firedrake import *

from config import *
from domain import *
from spaces import *


def viscosity(ux, uy, n=3.0):
    eps_e2 = ux.dx(0) ** 2 + ux.dx(1) ** 2

    mu = 0.5 * A**(-1.0 / n) * (eps_e2 + eps**2) ** ((1.0 - n) / (2.0 * n))
    return mu

