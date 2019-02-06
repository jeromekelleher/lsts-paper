"""
Direct implementations of the algorithms in the paper.
"""

import msprime
import numpy as np


def ls_matrix_n2(h, H, rho, theta):
    """
    Simplest implementation of LS viterbi algorithm. There's no normalisation
    of the V values here so this will fail for anything but the smallest
    examples.

    Complexity:

    Space = O(n m) (but in practise uses large V and T matrices)
    Time = O(m n^2)
    """
    n, m = H.shape
    r = 1 - np.exp(-rho / n)
    transition_proba = np.zeros((n, n))
    transition_proba[:,:] = r / n
    np.fill_diagonal(transition_proba, 1 - r + r / n)
    # We have two observations: different (0), and the same (1).
    # The emission probability is the same for each state.
    emission_proba = np.array([
        0.5 * theta / (n + theta),
        n / (n + theta) + 0.5 * theta / (n + theta)])
    V = np.zeros((m, n))
    T = np.zeros((m, n), dtype=int)

    for j in range(n):
        V[0, j] = (1 / n) * emission_proba[int(h[0] == H[j, 0])]
    for l in range(1, m):
        for j in range(n):
            max_p = 0
            max_k = -1
            for k in range(n):
                p = V[l - 1, k] * transition_proba[k, j]
                if p > max_p:
                    max_p = p
                    max_k = k
            V[l, j] = max_p * emission_proba[int(h[l] == H[j, l])]
            T[l, j] = max_k

    path = np.zeros(m, dtype=int)
    path[-1] = T[-1][np.argmax(V[-1])]
    for j in range(m - 1, 0, -1):
        path[j - 1] = T[j, path[j]]

    return path

def ls_matrix(h, H, rho, mu):
    """
    Simple matrix based method for LS Viterbi.

    Complexity:

    Space = O(n m) (Traceback matrix is still large)
    Time = O(n m)
    """
    # We must have a non-zero mutation rate, or we'll end up with
    # division by zero problems.
    assert np.all(mu > 0)

    n, m = H.shape
    V = np.ones(n)
    T = [set() for _ in range(m)]
    I = np.zeros(m, dtype=int)
    A = 2 # Fixing to binary for now.

    for l in range(m):
        I[l] = np.argmax(V)
        V /= V[I[l]]
        Vn = np.zeros(n)
        p_neq = rho[l] / n
        for j in range(n):
            p_t = (1 - rho[l] - rho[l] / n) * V[j]
            if p_neq > p_t:
                p_t = p_neq
                T[l].add(j)
            p_e = mu[l]
            if H[j, l] == h[l]:
                p_e = 1 - (A - 1) * mu[l]
            Vn[j] = p_t * p_e
        V = Vn
    # Traceback
    P = np.zeros(m, dtype=int)
    l = m - 1
    P[l] = np.argmax(V)
    while l > 0:
        j = P[l]
        if j in T[l]:
            j = I[l]
        P[l - 1] = j
        l -= 1
    return P


def in_sorted(values, j):
    # Take advantage of the fact that the numpy array is sorted.
    ret = False
    index = np.searchsorted(values, j)
    if index < values.shape[0]:
        ret = values[index] == j
    return ret


def ls_matrix_vectorised(h, H, rho, mu):
    # We must have a non-zero mutation rate, or we'll end up with
    # division by zero problems.
    assert np.all(mu > 0)

    n, m = H.shape
    V = np.ones(n)
    T = [None for _ in range(m)]
    I = np.zeros(m, dtype=int)
    A = 2 # Fixing to binary for now.

    for l in range(m):
        I[l] = np.argmax(V)
        V /= V[I[l]]
        # Transition
        p_neq = rho[l] / n
        p_t = (1 - rho[l] - rho[l] / n) * V
        recombinations = np.where(p_neq > p_t)[0]
        p_t[recombinations] = p_neq
        T[l] = recombinations
        # Emission
        p_e = np.zeros(n) + mu[l]
        index = H[:, l] == h[l]
        p_e[index] = 1 - (A - 1) * mu[l]
        V = p_t * p_e

    # Traceback
    P = np.zeros(m, dtype=int)
    l = m - 1
    P[l] = np.argmax(V)
    while l > 0:
        j = P[l]
        if in_sorted(T[l], j):
            j = I[l]
        P[l - 1] = j
        l -= 1
    return P


def ls_tree_naive(h, ts, rho, mu):
    """
    Simple tree based method of performing LS where we have a single tree.
    """
    assert ts.num_trees == 1
    assert ts.num_sites == len(h)
    V = np.zeros(ts.num_nodes) - 1
    tree = ts.first()
    V[tree.root] = 1
    for site in tree.sites():
        assert len(site.mutations) == 1




def main():
    np.set_printoptions(linewidth=1000)

    ts = msprime.simulate(25, recombination_rate=1, mutation_rate=2,
            random_seed=2, length=10)
    # ts = msprime.simulate(10, recombination_rate=0, mutation_rate=2, random_seed=2)
    H = ts.genotype_matrix().T
    print("Shape = ", H.shape)
    h = np.zeros(ts.num_sites, dtype=int)
    h[ts.num_sites // 2:] = 1
    # h = H[0]
    # h[ts.num_sites // 2:] = H[-1, ts.num_sites // 2:]


    r = 1
    # rho = np.zeros(ts.num_sites) + 1
    mu = np.zeros(ts.num_sites) + 1e-3
    np.random.seed(1)
    rho = np.random.random(ts.num_sites)
    # mu = np.random.random(ts.num_sites) #* 0.000001
    # path = ls_matrix(h, H, rho, mu)
    path = ls_matrix_vectorised(h, H, rho, mu)
    # assert np.all(path == path2)
    # print("p1", path)
    # print("p2", path2)
    match = H[path, np.arange(ts.num_sites)]
    print("h     = ", h)
    print("path  = ", path)
    print("match = ", match)
    print("eq    = ", np.all(h == match))

    # path = ls_tree_naive(h, ts, rho, mu)

if __name__ == "__main__":
    main()
