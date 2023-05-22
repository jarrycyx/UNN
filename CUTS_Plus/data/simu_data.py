import os
import sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
sys.path.append(opj(os.getcwd(), "../"))
sys.path.append(os.getcwd())

import csv
import tqdm
import torch
import scipy
from .generate_data_mod import generate_random_contemp_model, generate_nonlinear_contemp_timeseries
import numpy as np
from scipy.integrate import odeint
from einops import rearrange

from utils.spring_sim import SpringSim





def load_springs_data(T, N, interaction_strength=0.1, sample_freq=10, noise_std=0.1):    
    sim = SpringSim(interaction_strength=interaction_strength, n_balls=N, noise_var=noise_std**2)
    loc, vel, edges = sim.sample_trajectory(
        T=T*sample_freq,
        sample_freq=sample_freq,
        fixed_particle=False,
        influencer=False,
        uninfluenced=False,
        confounder=False,
    )  # 5000, 100

    true_cm = edges
    true_cm += np.eye(true_cm.shape[0])
    print(true_cm)
    
    X_np = rearrange(loc, "t d n -> t n d")
    return X_np, edges





######################################
# Function for loading input data 
######################################
def loadTrainingData(inputDataFilePath, device):

    # Load and parse input data (create batch data)
    inpData = torch.load(inputDataFilePath)
    Xtrain = torch.zeros(inpData['TsData'].shape[1], inpData['TsData'].shape[0], requires_grad = False, device=device)
    Xtrain1 = inpData['TsData'].t()
    Xtrain.data[:,:] = Xtrain1.data[:,:]

    return Xtrain

#######################################################
# Function for reading ground truth network from file 
#######################################################
def loadTrueNetwork(inputFilePath, networkSize):

    with open(inputFilePath) as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')
        numrows = 0    
        for row in reader:
            numrows = numrows + 1

    network = np.zeros((numrows,2),dtype=np.int16)
    with open(inputFilePath) as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')
        rowcounter = 0
        for row in reader:
            network[rowcounter][0] = int(row[0][1:])
            network[rowcounter][1] = int(row[1][1:])
            rowcounter = rowcounter + 1 

    Gtrue = np.zeros((networkSize,networkSize), dtype=np.int16)
    for row in range(0,len(network),1):
        Gtrue[network[row][1]-1][network[row][0]-1] = 1   
    
    return Gtrue


def load_dream_data(dataset_id):
    device = "cpu"

    if(dataset_id == 0):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli1.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli1.tsv"
    elif(dataset_id == 1):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli2.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv"
    elif(dataset_id == 2):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast1.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast1.tsv"
    elif(dataset_id == 3):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast2.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast2.tsv"
    elif(dataset_id == 4):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast3.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast3.tsv"
    elif(dataset_id == 5):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size10Ecoli1.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Ecoli1.tsv"
    elif(dataset_id == 6):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size10Ecoli2.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Ecoli2.tsv"
    elif(dataset_id == 7):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size10Yeast1.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Yeast1.tsv"
    elif(dataset_id == 8):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size10Yeast2.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Yeast2.tsv"
    elif(dataset_id == 9):
        InputDataFilePath = "SRU_for_GCI/data/dream3/Dream3TensorData/Size10Yeast3.pt"
        RefNetworkFilePath = "SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Yeast3.tsv"
    else:
        print("Error while loading gene training data")    

    Xtrain = loadTrainingData(InputDataFilePath, device)
    n = Xtrain.shape[0]
    Gref = loadTrueNetwork(RefNetworkFilePath, n)
    
    Xtrain = Xtrain.numpy().T
    # Gref = Gref.T
    
    return Xtrain, Gref





def load_netsim_data(dataset_id):
    fileName = "SRU_for_GCI/data/netsim/sim3_subject_%s.npz" % (dataset_id)
    ld = np.load(fileName)
    X_np = ld['X_np']
    X_np = rearrange(X_np, "n t -> t n")
    Gref = ld['Gref']
    return X_np, Gref





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


def simulate_random_var(seed, T, N, L, coef=[0.2, 0.8], auto_corr=[0.4,0.9], tau_max=5, noise_sigma=[0.01, 0.01]):

    if True:
        coupling_funcs = [lin_f]
        noise_types = ['gaussian']  # , 'weibull', 'uniform']
        # noise_sigma = (0.1, 0.3)

    couplings = list(np.arange(coef[0], coef[1]+1e-5, coef[2]))
    couplings += [-c for c in couplings]

    # auto_deps = list(np.arange(max(0., auto_corr-0.6), auto_corr+0.01, 0.05))
    auto_deps = list(np.arange(auto_corr[0], auto_corr[1]+1e-5, auto_corr[2]))

    # Models may be non-stationary. Hence, we iterate over a number of seeds
    # to find a stationary one regarding network topology, noises, etc

    ir = 0
    model_seed = seed
    while True:
        ir += 1
        # np.random.seed(model_seed)
        random_state = np.random.RandomState(model_seed)

        links = generate_random_contemp_model(
            N=N, L=L,
            coupling_coeffs=couplings,
            coupling_funcs=coupling_funcs,
            auto_coeffs=auto_deps,
            tau_max=tau_max,
            contemp_fraction=0.,
            # num_trials=1000,
            random_state=random_state)

        noises = []
        for j in links:
            noise_type = random_state.choice(noise_types)
            sigmas = list(np.arange(noise_sigma[0], noise_sigma[1]+1e-5, noise_sigma[2]))
            sigma = random_state.choice(sigmas)
            # sigma = noise_sigma[0] + (noise_sigma[1]-noise_sigma[0])*random_state.rand()
            noises.append(getattr(noise_model(sigma=sigma, seed=seed), noise_type))

        data_all_check, nonstationary = generate_nonlinear_contemp_timeseries(
            links=links, T=100, noises=noises, random_state=random_state)

        # If the model is stationary, break the loop
        if not nonstationary:
            data, nonstationary_full = generate_nonlinear_contemp_timeseries(
                links=links, T=T, noises=noises, random_state=random_state)
            if not nonstationary_full:
                break
        else:
            print("Trial %d: Not a stationary model" % ir)
            model_seed += 10000

    cm = links_to_matrix(links)
    return data, cm


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
    data, nonstationary = generate_nonlinear_contemp_timeseries(
        links=new_links, T=T, noises=noises, random_state=random_state)
    if nonstationary:
        print("Model nonstationay!")

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
        # print(f"Nonstationary, beta={str(beta):s}, max_eig={max_eig:.4f}")
        return make_var_stationary((beta / max_eig) * 0.7, radius)
    else:
        # print(f"Stationary, beta={str(beta):s}")
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
        
    data = X.T[burn_in:, :]
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

    return X[burn_in:, :], GC

