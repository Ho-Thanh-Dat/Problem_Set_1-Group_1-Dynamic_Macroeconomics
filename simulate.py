# simulate.py
import numpy as np
from numpy import cumsum,linspace,squeeze,where,zeros
from numpy.random import choice,rand,seed
from numpy.linalg import matrix_power
from types import SimpleNamespace

def grow_economy(planner_instance, sol):
    par = planner_instance.par

    # Unpack parameters
    T = par.T
    kgrid = par.kgrid
    Agrid = par.Agrid.flatten()  # Ensure Agrid is 1-dimensional
    pmat = par.pmat
    hgrid = par.hgrid

    # Initial conditions
    k0 = par.k0
    A0 = par.A0
    H0 = par.H0

    # Find initial indices
    k0_ind = np.argmin(np.abs(kgrid - k0))
    A0_ind = np.argmin(np.abs(Agrid - A0))
    H0_ind = np.argmin(np.abs(hgrid - H0))

    # Pre-allocate arrays for simulation
    ksim = np.zeros(T)
    Asim = np.zeros(T)
    Hsim = np.zeros(T)
    ysim = np.zeros(T)
    n_sim = np.zeros(T)
    csim = np.zeros(T)  # Initialize csim
    isim = np.zeros(T)  # Initialize isim
    usim = np.zeros(T)  # Initialize usim

    # Initial values
    ksim[0] = k0
    Asim[0] = A0
    Hsim[0] = H0

    # Simulate the economy
    for t in range(1, T):
        # Get indices for current state
        k_ind = np.argmin(np.abs(kgrid - ksim[t-1]))
        A_ind = np.argmin(np.abs(Agrid - Asim[t-1]))
        H_ind = np.argmin(np.abs(hgrid - Hsim[t-1]))

        # Ensure indices are within bounds
        k_ind = np.clip(k_ind, 0, par.klen - 1)
        A_ind = np.clip(A_ind, 0, par.Alen - 1)
        H_ind = np.clip(H_ind, 0, par.hlen - 1)

        # Get policy functions
        k_next = sol['k'][k_ind, A_ind, H_ind]
        u_next = sol['u'][k_ind, A_ind, H_ind]  # Use 'u' instead of 'n'
        y_next = sol['y'][k_ind, A_ind, H_ind] # Get the output
        c_next = sol['c'][k_ind, A_ind, H_ind]
        i_next = sol['i'][k_ind, A_ind, H_ind]
        utility_next = par.util(c_next, u_next, par.sigma, par.nu, par.gamma) #consistent naming

        # Update state variables
        ksim[t] = k_next
        Asim[t] = np.random.choice(Agrid, p=pmat[A_ind, :])
        Hsim[t] = (1 - par.delta_H) * Hsim[t-1] + par.s_h * y_next
        ysim[t] = y_next
        n_sim[t] = u_next
        csim[t] = c_next  # Store simulated consumption
        isim[t] = i_next  # Store simulated investment
        usim[t] = utility_next  # Store simulated utility

    return {
        'ksim': ksim,
        'Asim': Asim,
        'Hsim': Hsim,
        'ysim': ysim,
        'n_sim': n_sim,
        'usim' : usim,
        'isim' : isim,
        'csim' : csim
    }