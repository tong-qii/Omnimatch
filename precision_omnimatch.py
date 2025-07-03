# This code produces the precision plots of the correctly unshuffled vertices for S-OmniMatch (Figure 9) 
# with varying number of shuffled vertices, embedding dimensions, and number of neighbors used in S-OmniMatch.

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
from graspologic.inference import latent_distribution_test
from graspologic.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspologic.simulations import sbm, rdpg
from graspologic.utils import symmetrize
from graspologic.plot import heatmap, pairplot
import igraph as ig
from igraph import *
from scipy.sparse import csr_array
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

def array_split(data, m):
    rowIdx = np.arange(data.shape[0])
    # Split row indices into m groups
    split_indices = np.array_split(rowIdx, m)
    
    # Use the split indices to split the data array
    split_data = [data[x, :] for x in split_indices]
    
    return split_data

def check_repeated_rows(df):
    # Use duplicated() method to check for repeated rows
    duplicated_rows = df[df.duplicated()]
    if not duplicated_rows.empty:
        print("There are repeated rows.")
        print(duplicated_rows)
        print('numer is',duplicated_rows.shape )
    else:
        print("No repeated rows.")
        
# this is where k is changed
def find_smallest_columns(row, num): # edit the k for softmatching
    return np.array(row.nsmallest(num).index)

def soft_matching_estimate(dists, oosB, vertex_shuffled, dim, num):
    # This function computes the average latent positions for S-OmniMatch
    dists_df = pd.DataFrame(dists)
    dists_df.columns = [vertex_shuffled] #vertex_shuffled2
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

def precision_plot(n, kshuff, nMC, dims_to_run, step):
    # This function produces the precision plot

    num = np.arange(0, kshuff + 1, step)
    num[0] = 1
    results = {}

    for dim in dims_to_run:
        prop_corr_soft = np.zeros((nMC, len(num)))

        for h in range(nMC):
            X = np.random.dirichlet(np.ones(dim + 2), n)
            X = np.delete(X, [dim, dim + 1], axis=1)
            adj1 = rdpg(X)
            adj1.astype(int)

            order_original = np.arange(0, n)
            vertex_shuffled = list(reversed(np.arange(n - kshuff, n)))
            unshuf = [item for item in order_original if item not in vertex_shuffled]
            reorder_vertex = np.concatenate((unshuf, vertex_shuffled))

            X2 = X[reorder_vertex, :]
            adj2 = rdpg(X2)
            adj2.astype(int)

            MA = adj1[unshuf, :][:, unshuf]
            MB = adj2[0:(n - kshuff), :][:, 0:(n - kshuff)]

            omni_embedder = OmnibusEmbed(n_components=dim)
            Zhats12 = omni_embedder.fit_transform([MA, MB])
            MA_xhat = Zhats12[0]
            MB_xhat = Zhats12[1]

            oosA = np.zeros((kshuff, dim))
            oosB = np.zeros((kshuff, dim))

            for i in range(kshuff):
                shuffle_vector = adj1[n - kshuff + i, :(n - kshuff)]
                oosA[i, :] = aseoos(MA_xhat, shuffle_vector)

                shuffle_vector = adj2[n - kshuff + i, :(n - kshuff)]
                oosB[i, :] = aseoos(MB_xhat, shuffle_vector)

            for j in range(len(num)):
                dists = cdist(oosA, oosB, 'euclidean')
                num_best_guesses = np.argsort(dists, axis=1)[:, :num[j]]
                undo_shuff = list(reversed(np.arange(kshuff)))
                matches = np.array([undo_shuff[i] in num_best_guesses[i] for i in np.arange(kshuff)])
                prop_corr_soft[h, j] = np.mean(matches)

        df_soft = pd.DataFrame(np.vstack((num, prop_corr_soft)).T)
        df_soft.columns = ["num"] + list(np.arange(nMC))
        df_soft['mean'] = df_soft.iloc[:, 1:(nMC + 1)].mean(axis=1)
        df_soft['SE'] = np.sqrt(df_soft['mean'] * (1 - df_soft['mean']) / np.sqrt(nMC))

        results[dim] = df_soft

    # Plotting the figure
    plt.figure(figsize=(10, 6))
    for dim in dims_to_run:
        df = results[dim]
        plt.errorbar(df['num'], df['mean'], yerr=df['SE'], fmt='o-', capsize=5, label=f'd = {dim}', markersize=4)

    plt.xlabel("Number of Shuffled Vertices")
    plt.ylabel("Proportion Correctly Recovered")
    plt.title('Proportion of recovered vertices after shuffling \n on RDPG, n = ' + str(n) + 
              ', nMC = '+str(nMC) + ', n_s = ' + str(kshuff))
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'new_precision_n{n}_nMC{nMC}_dims{"-".join(map(str, dims_to_run))}_k{kshuff}.png')

    # Saving results
    save_dict = {
        'kshuff': kshuff,
        'results': results,
        'n': n,
        'num': num,
        'nMC': nMC,
        'dims_to_run': dims_to_run
    }
    
    # Saving figure
    np.savez(f'precision_results_n{n}_nMC{nMC}_dims{"-".join(map(str, dims_to_run))}_k{kshuff}.npz', **save_dict)

    return results

# To load saved results later (example code):
    # data = np.load('precision_results_n10000_nMC100_dims2-10-15_k50.npz', allow_pickle=True)
    # kshuff = data['kshuff']
    # results = data['results']

# Example usage:
# precision_plot(n=10000, kshuff=500, nMC=100, dims_to_run=[2, 10, 15], step=50)