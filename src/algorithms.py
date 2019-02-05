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
    # TODO set the mutation and recombination rates appropriately following
    # other examples in the literature.

    n, m = H.shape
    V = np.ones(n)
    T = np.zeros((n, m), dtype=int)
    A = 2 # Fixing to binary for now.

    for l in range(m):
        max_V_index = np.argmax(V)
        V /= V[max_V_index]
        Vn = np.zeros(n)
        for j in range(n):
            x = (1 - rho[l] - rho[l] / n) * V[j]
            y = rho[l] / n
            if x > y:
                p_t = x
                T[j, l] = j
            else:
                p_t = y
                T[j, l] = max_V_index
            if H[j, l] == h[l]:
                p_e = 1 - (A - 1) * mu[l]
            else:
                p_e = mu[l]
            Vn[j] = p_t * p_e
        V = Vn
    # Traceback
    P = np.zeros(m, dtype=int)
    P[m - 1] = max_V_index
    for l in range(m - 1, 0, -1):
        P[l - 1] = T[P[l], l]
    return P


def main():
    np.set_printoptions(linewidth=1000)

    ts = msprime.simulate(10, recombination_rate=1, mutation_rate=2, random_seed=2)
    H = ts.genotype_matrix().T
    print(H)
    h = np.zeros(ts.num_sites, dtype=int)
    h[5:] = 1

    r = 1
    rho = np.zeros(ts.num_sites) + 1 - np.exp(r / ts.num_samples)
    mu = np.zeros(ts.num_sites) + 0.01
    path = ls_matrix(h, H, rho, mu)

    match = H[path, np.arange(ts.num_sites)]
    print("h     = ", h)
    print("path  = ", path)
    print("match = ", match)

if __name__ == "__main__":
    main()
