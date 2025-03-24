# solve.py
from numpy import argmax, expand_dims, inf, squeeze, tile, zeros, seterr, array, where, log
from numpy.linalg import norm
from scipy.optimize import fminbound, minimize_scalar  # Use minimize_scalar for guaranteed bounds
from types import SimpleNamespace
import time
import numpy as np

seterr(all='ignore')

def plan_allocations(planner_instance):
    print('Solving the Model by Value Function Iteration')
    sol = {}
    myClass = planner_instance
    par = myClass.par

    beta = par.beta
    alpha = par.alpha
    delta = par.delta
    sigma = par.sigma
    eta = par.eta
    delta_H = par.delta_H
    s_h = par.s_h

    klen = par.klen
    kgrid = par.kgrid

    Alen = par.Alen
    Agrid = par.Agrid.flatten()  # Ensure Agrid is 1-dimensional
    pmat = par.pmat

    hlen = par.hlen
    hgrid = par.hgrid

    kmat = np.tile(kgrid.reshape(klen, 1, 1), (1, Alen, hlen))
    Amat = np.tile(Agrid.reshape(1, Alen, 1), (klen, 1, hlen))
    Hmat = np.tile(hgrid.reshape(1, 1, hlen), (klen, Alen, 1))

    util = par.util

    v0 = np.zeros((klen, Alen, hlen))
    v1 = np.zeros((klen, Alen, hlen))
    k1 = np.zeros((klen, Alen, hlen))
    h1 = np.zeros((klen, Alen, hlen))
    u1 = np.zeros((klen, Alen, hlen))
    y1 = np.zeros((klen, Alen, hlen))
    i1 = np.zeros((klen, Alen, hlen))
    c1 = np.zeros((klen, Alen, hlen))

    t0 = time.time()

    print('Iterating on Bellman Eq.')

    y0 = Amat * (kmat**alpha) * ((2 - delta_H) * Hmat)**eta
    i0 = -(1 - delta) * kmat
    c0 = y0 - i0
    v0 = util(c0, 1, sigma, par.nu, par.gamma) / (1.0 - beta)

    crit = 1e-6
    maxiter = 10000
    diff = 1
    iter = 0

    while diff > crit and iter < maxiter:
        for j in range(Alen):
            ev = np.zeros((klen, hlen))
            for pp in range(klen):
                for qq in range(hlen):
                    ev[pp, qq] = np.dot(v0[pp, :, qq], pmat[j, :])

            for p in range(klen):
                for q in range(hlen):
                    u = 2.0 - delta_H - (hgrid / hgrid[q])
                    hub = np.where(u > 0.0)[0][-1]
                    hlb = np.where(u <= 1.0)[0][0]

                    hp = hgrid.copy()
                    hp[u > 1.0] = hgrid[hlb]
                    hp[u <= 0.0] = hgrid[hub]
                    u = 2.0 - delta_H - (hp / hgrid[q])

                    y = Agrid[j] * (kgrid[p]**alpha) * (u * hgrid[q])**eta
                    i = kgrid - (1 - delta) * kgrid[p]
                    c = y - i[:, np.newaxis]  # Ensure shapes are compatible
                    c[c < 0.0] = 0.0

                    vall = util(c, 1, sigma, par.nu, par.gamma) + beta * ev
                    vall[c <= 0.0] = -np.inf
                    vmax = np.max(vall)
                    ind = np.unravel_index(np.argmax(vall, axis=None), vall.shape)
                    kind, hind = ind

                    v1[p, j, q] = vmax
                    k1[p, j, q] = kgrid[kind]
                    u1[p, j, q] = 1 - u[hind]
                    h1[p, j, q] = hp[hind]
                    i1[p, j, q] = i[kind]
                    c1[p, j, q] = c[kind, hind]
                    y1[p, j, q] = y[hind]

        diff = norm(v1 - v0)
        v0 = v1.copy()

        iter += 1
        if iter % 25 == 0:
            print(f'Iteration: {iter}, Difference: {diff:.7f}')

    t1 = time.time()
    print(f'Elapsed time is {t1 - t0:.2f} seconds.')
    print(f'Converged in {iter} iterations.')

    sol['y'] = y1
    sol['k'] = k1
    sol['h'] = h1
    sol['i'] = i1
    sol['u'] = u1
    sol['c'] = c1
    sol['v'] = v1

    return sol

def intra_foc(n, k, kp, A, h, alpha, eta, delta, sigma, nu, gamma):
    c = (A * (k**alpha) * (h**eta) * (n**(1.0 - alpha - eta))) + ((1.0 - delta)*k - kp)
    c = np.maximum(c, 1e-8)
    mpl = A * (1.0 - alpha - eta) * (k**alpha) * (h**eta) * (n**(-alpha - eta))
    un = -gamma * (1.0 - n)**(1.0 / nu)
    if sigma == 1.0:
        uc = 1.0 / c
    else:
        uc = c**(-sigma)
    return -(uc * mpl + un)