# This code produces the matching plots comparing the proportion of correctly recovered vertices after shuffling
# for OmniMatch and S-OmniMatch (Figure 2 and Figure 8) with varying number of shuffled vertices, embedding dimensions, 
# and number of neighbors used in S-OmniMatch. 
#
# The code is written in parallel for faster computation.

# Importing packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import networkx as nx
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import cdist
import lap
from numpy import linalg as LA
from multiprocessing import cpu_count
from sklearn.metrics.pairwise import pairwise_kernels
import graspologic
from graspologic.embed import OmnibusEmbed
from graspologic.simulations import rdpg
import igraph as ig
from igraph import *
from scipy.sparse import csr_array
from joblib import Parallel, delayed
import psutil
import os
np.random.seed(8888)

# Defining functions
def aseoos(Xhat, avec):
    # linear least squares solver (for estimating latent positions in embedding, Xhat)
    # out of sample embedding
    n = Xhat.shape[0]
    
    if len(avec) != n:
        raise ValueError("Length of edge vector avec does not match size of latent position matrix Xhat.")
    
    # solve Xhat @ ooshat = avec
    # (really we're solving the linear least squares)
    ooshat = np.linalg.lstsq(Xhat.T @ Xhat, Xhat.T @ avec, rcond=None)[0]
    
    return ooshat

def find_smallest_columns(row, num):
    return np.array(row.nsmallest(num).index)

def soft_matching_estimate(dists, oosB, vertex_shuffled, dim, num):
    # This function computes the average latent positions for S-OmniMatch

    dists_df = pd.DataFrame(dists)
    dists_df.columns = [vertex_shuffled] 
    dists_df.index = [vertex_shuffled]  
    smallest_columns = dists_df.apply(lambda x: find_smallest_columns(x, num), axis=1)
    flattened_5cols = []
    for row in smallest_columns:
        flattened_5cols.append([item[0] for item in row])
    oosB_df = pd.DataFrame(oosB)
    oosB_df.index = [vertex_shuffled] 
    oosB_avg = np.zeros((len(vertex_shuffled), dim))
    for i in range(len(vertex_shuffled)):
        avg = np.array(oosB_df.loc[flattened_5cols[i]].mean())
        oosB_avg[i,:] = avg
    return oosB_avg

def run_one_MC(h, n, dim, num, kshuff):
    # This function performs one MC simulation in parallel

    prop_corr = np.zeros(len(kshuff))
    prop_corr_soft_dict = {k: np.zeros(len(kshuff)) for k in num}

    for j in range(len(kshuff)):
        X = np.random.dirichlet(np.ones(dim + 2), n)
        X = np.delete(X, [dim, dim + 1], axis=1)
        adj1 = rdpg(X)
        adj1.astype(int)

        order_original = np.arange(0, n)
        vertex_shuffled = list(reversed(np.arange(n-kshuff[j], n)))
        true_positions = list(np.arange(n - kshuff[j], n))
        unshuf = [item for item in order_original if item not in vertex_shuffled]
        reorder_vertex = np.concatenate((unshuf, vertex_shuffled))

        X2 = X[reorder_vertex, :]
        adj2 = rdpg(X2)
        adj2.astype(int)

        MA = adj1[unshuf, :][:, unshuf]
        MB = adj2[0:(n-kshuff[j]), :][:, 0:(n-kshuff[j])]

        omni_embedder = OmnibusEmbed(n_components=dim)
        Zhats12 = omni_embedder.fit_transform([MA, MB])
        MA_xhat = Zhats12[0]
        MB_xhat = Zhats12[1]

        oosA = np.zeros((kshuff[j], dim))
        for i in range(kshuff[j]):
            shuffle_vector = adj1[n-kshuff[j]+i, :(n-kshuff[j])]
            oosA[i, :] = aseoos(MA_xhat, shuffle_vector)

        oosB = np.zeros((kshuff[j], dim))
        for i in range(kshuff[j]):
            shuffle_vector = adj2[n-kshuff[j]+i, :(n-kshuff[j])]
            oosB[i, :] = aseoos(MB_xhat, shuffle_vector)

        dists = cdist(oosA, oosB, 'euclidean')
        _, x1, _ = lap.lapjv(dists)

        correct_matches = np.sum(true_positions == np.take(vertex_shuffled, x1))
        prop_corr[j] = correct_matches / kshuff[j]

        for k in num:
            num_best_guesses = np.argsort(dists, axis=1)[:, :k]
            undo_shuff = list(reversed(np.arange(kshuff[j])))
            matches = np.array([undo_shuff[i] in num_best_guesses[i] for i in np.arange(kshuff[j])])
            prop_corr_soft_dict[k][j] = np.mean(matches)

    return prop_corr, prop_corr_soft_dict

def prop_plot(n, dim, num, nMC=100):
    # This function plots the results of the nMC number of MC simulations

    kshuff = int(n / 500) * np.array([20, 30, 40, 50, 60, 80, 90, 100, 110, 120])

    # Results from parallel computing
    results = Parallel(n_jobs=10, verbose=10)(
        delayed(run_one_MC)(h, n, dim, num, kshuff) for h in range(nMC)
    )

    prop_corr = np.array([res[0] for res in results])
    prop_corr_soft_dict = {k: np.array([res[1][k] for res in results]) for k in num}

    df = pd.DataFrame(np.vstack((kshuff, prop_corr)).T)
    df.columns = ["kshuff"] + list(np.arange(nMC))
    df['mean'] = df.iloc[:, 1:(nMC+1)].mean(axis=1)
    df['SE'] = np.sqrt(df['mean'] * (1 - df['mean']) / np.sqrt(nMC))

    plt.figure(figsize=(8, 5))
    plt.errorbar(kshuff, df['mean'], yerr=df['SE'], fmt='o-', 
                 capsize=5, label='Omni-OOS', markersize=5)

    for k in num:
        df_soft = pd.DataFrame(np.vstack((kshuff, prop_corr_soft_dict[k])).T)
        df_soft.columns = ["kshuff"] + list(np.arange(nMC))
        df_soft['mean'] = df_soft.iloc[:, 1:(nMC+1)].mean(axis=1)
        df_soft['SE'] = np.sqrt(df_soft['mean'] * (1 - df_soft['mean']) / np.sqrt(nMC))

        plt.errorbar(kshuff, df_soft['mean'], yerr=df_soft['SE'], fmt='s-', 
                     capsize=5, label=f'S-OmniMatch (k={k})', markersize=5)

    plt.xlabel("Number of Shuffled Vertices")
    plt.ylabel("Proportion Correctly Recovered")
    plt.title('Proportion of recovered vertices after shuffling \n on RDPG, n = ' + str(n) + 
              ', nMC = '+str(nMC)+ ', d = '+str(dim))       
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'new_prop_n{n}_nMC{nMC}_d{dim}_ks{"-".join(map(str, num))}.png')

    # Saving the results
    save_dict = {
        'kshuff': kshuff,
        'prop_corr': prop_corr,
        'prop_corr_soft_dict': prop_corr_soft_dict,
        'n': n,
        'dim': dim,
        'num': num,
        'nMC': nMC
    }

    # Saving the figure 
    np.savez(f'new_prop_results_n{n}_d{dim}_nMC{nMC}_k{"-".join(map(str, num))}.npz', **save_dict)

# To load saved results later (example code):
    # data = np.load('prop_results_n1000_d2_nMC100_k1-5-10.npz', allow_pickle=True)
    # kshuff = data['kshuff']
    # prop_corr = data['prop_corr']
    # prop_corr_soft_dict = data['prop_corr_soft_dict'].item()  # `.item()` because it's a dict

# Example usage:
# prop_plot(n=500, dim = 2, num = [1,3,5,10])
