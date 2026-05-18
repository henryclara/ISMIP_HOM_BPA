from firedrake import *

def viscosity(ux, uy, n=1):
    '''
    Double check this against the derivation
    '''
    eps_e2 = (ux.dx(0)**2 + uy.dx(1)**2 + ux.dx(0) * uy.dx(1) \
              + 0.25 * (ux.dx(1) + uy.dx(0))**2 + 0.25 * ux.dx(2)**2 \
              + 0.25 * uy.dx(2)**2)

    mu = 0.5 * A**(-1.0 / n) * (eps_e2 + eps**2)**((1.0 - n) / (2.0 * n))
    return mu

