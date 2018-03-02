import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import scipy.linalg as linalg



def rmANOVA(df):
    '''
    My attempt at a repeated-measures ANOVA 
    In:
        data: dataframe

    Out:
        x: symmetric matrix of f-statistics
        **coming soon** p: p-values for each element of x
    '''

    stats_dict = dict()
    for dset in df['dataset'].unique():
        alg_list = []
        for alg in df['pipeline'].unique():
            alg_list.append(df[np.logical_and(
                df['dataset'] == dset, df['pipeline'] == alg)]['score'].as_matrix())
        alg_list = [a for a in alg_list if len(a) > 0] #some datasets and algorithms may not exist?
        M = np.stack(alg_list).T
        stats_dict[dset] = _rmanova(M)
    return stats_dict


def _rmanova(matrix):
    mean_subj = matrix.mean(axis=1)
    mean_algo = matrix.mean(axis=0)
    grand_mean = matrix[:].mean()

    # SS: sum of squared difference
    SS_algo = len(mean_subj) * np.sum((mean_algo - grand_mean)**2)
    SS_within_subj = np.sum((matrix - mean_algo[np.newaxis, :])**2)
    SS_subject = len(mean_algo) * np.sum((mean_subj - grand_mean)**2)
    SS_error = SS_within_subj - SS_subject

    # MS: Mean of squared difference
    MS_algo = SS_algo / (len(mean_algo) - 1)
    MS_error = SS_error / ((len(mean_algo) - 1)*(len(mean_subj) - 1))

    # F-statistics
    f = MS_algo/MS_error
    n, k = matrix.shape
    df1 = k-1
    df2 = (k-1)*(n-1)  # calculated as one-way repeated-measures ANOVA
    p = stats.f.sf(f, df1, df2)
    return f, p

def interleave_vectors(a,b):
    out = np.zeros((len(a)+len(b),1))
    out[0::2,0] = a
    out[1::2,0] = b
    return out

def generate_perm_matrix(b):
    return linalg.block_diag(*[np.array([1,-1]) if not x else np.array([-1,1]) for x in b])

def return_null_distribution(x, y, iterations):
    xy = interleave_vectors(x,y)
    M = np.stack([generate_perm_matrix(np.random.binomial(1,0.5,len(y))) for i in range(iterations)])
    return np.squeeze(np.tensordot(M,xy,axes=[[2],[0]]))

def permutation_pairedttest(x, y, iterations=1000):
    null = return_null_distribution(x, y, iterations).mean(axis=1)
    true = (y-x).mean()
    return (null > true).mean()
    

