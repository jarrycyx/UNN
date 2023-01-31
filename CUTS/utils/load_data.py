
import os
import sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
sys.path.append(opj(os.getcwd(), "../"))
sys.path.append(os.getcwd())

import csv
import torch
import scipy
import utils.generate_data_mod as mod
import numpy as np
from einops import rearrange
from scipy.integrate import odeint


def links_to_matrix(links):
    N = len(links)
    cm = np.zeros([N, N])
    for i, effect_node in links.items():
        for (j, _), _, _ in effect_node:
            cm[i, j] += 1
    return cm


class noise_model:
    def __init__(self, sigma=1, seed=0):
        self.random_state = np.random.RandomState(seed)
        self.sigma = sigma

    def gaussian(self, T):
        # Get zero-mean unit variance gaussian distribution
        return self.sigma*self.random_state.randn(T)

    def weibull(self, T):
        # Get zero-mean sigma variance weibull distribution
        a = 2
        mean = scipy.special.gamma(1./a + 1)
        variance = scipy.special.gamma(
            2./a + 1) - scipy.special.gamma(1./a + 1)**2
        return self.sigma*(self.random_state.weibull(a=a, size=T) - mean)/np.sqrt(variance)

    def uniform(self, T):
        # Get zero-mean sigma variance uniform distribution
        mean = 0.5
        variance = 1./12.
        return self.sigma*(self.random_state.uniform(size=T) - mean)/np.sqrt(variance)


def lin_f(x): return x
def f2(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))


def simulate_var_from_links(links, T, seed=0, noise_sigma=[0.1, 0.2], noise_type="gaussian", func_name="lin_f"):
    """
    links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
                    1: [((1, -1), 0.8), ((3, -1), 0.8)],
                    2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
                    3: [((3, -1), 0.4)],
                    }
    """
    def get_func(func_name):
        if func_name == "lin_f":
            return lin_f
        else:
            raise NotImplementedError

    random_state = np.random.RandomState(seed)
    noises = []

    new_links = {}
    for j in range(len(links)):
        sigma = noise_sigma[0] + \
            (noise_sigma[1]-noise_sigma[0])*random_state.rand()
        noises.append(getattr(noise_model(sigma=sigma, seed=seed), noise_type))
        new_links[j] = []
        for props in links[j]:
            new_links[j].append(
                (tuple(props[0:2]), props[2], get_func(props[3]),))
    data, nonstationary = mod.generate_nonlinear_contemp_timeseries(
        links=new_links, T=T, noises=noises, random_state=random_state)
    if nonstationary:
        print("Model nonstationay!")

    data = data[:, :, None]
    cm = links_to_matrix(new_links)
    return data, cm






def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta



def load_netsim_data(dataset_id):
    fileName = "netsim_dataset/sim3_subject_%s.npz" % (dataset_id)
    ld = np.load(fileName)
    X_np = ld['X_np']
    X_np = rearrange(X_np, "n t -> t n 1")
    Gref = ld['Gref']
    return X_np, Gref




def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, auto_corr=3.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * auto_corr

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(loc=0, scale=sd, size=(p, T + burn_in))
    X = np.ones((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t-1]
        
    data = X.T[burn_in:, :, None]
    return data, beta, GC



def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt

def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


def simulate_lorenz_96_process(N, T, seed, F=10, sd=0.1):
    data, true_cm = simulate_lorenz_96(p=N, T=T, seed=seed, F=F, sd=sd)
    data = data[:, :, None]
    return data, true_cm