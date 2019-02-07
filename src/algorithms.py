"""
Direct implementations of the algorithms in the paper.
"""

import msprime
import tskit
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


def ls_matrix(h, H, rho, mu, V_matrix):
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
        V_matrix[l] = V
        I[l] = np.argmax(V)
        print("Site ", l, "maxV = ", V[I[l]])
        V /= V[I[l]]
        p_neq = rho[l] / n
        for j in range(n):
            p_t = (1 - rho[l] - rho[l] / n) * V[j]
            if p_neq > p_t:
                p_t = p_neq
                T[l].add(j)
            p_e = mu[l]
            if H[j, l] == h[l]:
                p_e = 1 - (A - 1) * mu[l]
            V[j] = p_t * p_e
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



def is_descendant(tree, u, v):
    """
    Returns True if the specified node u is a descendent of node v. That is,
    v is on the path to root from u.
    """
    ret = False
    if v != -1:
        w = u
        path = []
        while w != v and w != tskit.NULL:
            path.append(w)
            w = tree.parent(w)
        # print("DESC:",v, u, path)
        ret = w == v
    # print("IS_DESCENDENT(", u, v, ") = ", ret)
    return ret


def ls_tree_naive(h, ts, rho, mu, V_matrix):
    """
    Simple tree based method of performing LS where we have a single tree.
    """
    Lambda = -1
    assert ts.num_trees == 1
    assert ts.num_sites == len(h)
    I = np.zeros(ts.num_sites, dtype=int)
    T = [{} for _ in range(ts.num_sites)]
    L = np.zeros(ts.num_nodes) + Lambda
    A = 2 # Must be biallelic
    n, m  = ts.num_samples, ts.num_sites
    tree = ts.first()
    U = set()
    for u in ts.samples():
        L[u] = 1
        U.add(u)
    for site in tree.sites():
        assert len(U) > 0
        assert len(site.mutations) == 1
        mutation = site.mutations[0]

        V = np.zeros(n)
        for j in range(n):
            v = j
            while L[v] == Lambda:
                v = tree.parent(v)
            assert v != tskit.NULL
            V[j] = L[v]
        V_matrix[site.id] = V
        # print(site.id, h[site.id], V)

        # Add an L node for the mutation
        # u = mutation.node
        # print("Adding mutation", u, "to ", U)
        # while L[u] == Lambda:
        #     u = tree.parent(u)
        # L[mutation.node] = L[u]
        # U.add(mutation.node)

        l = site.id
        # print(l, U, L)
        # Find max likelihood and normalise
        I[l] = max(U, key=lambda u: L[u])
        max_L = L[I[l]]

        print("Site", l, "max_l = ", max_L)
        node_labels = {u: "{}:{:.6g}".format(u, L[u]) for u in U}
        for u in tree.samples():
            if u not in node_labels:
                node_labels[u] = "{}      ".format(u)
        print(tree.draw(format="unicode", node_labels=node_labels))
        print("mutation = ", mutation.node, "state = ", h[l])
        for u in U:
            L[u] /= max_L

        p_neq = rho[l] / n
        for u in U:
            p_t = (1 - rho[l] - rho[l] / n) * L[u]
            T[l][u] = False
            if p_neq > p_t:
                p_t = p_neq
                T[l][u] = True
            # Assuming single polarised 0/1 mutation
            d = is_descendant(tree, u, mutation.node)
            state = int(d)
            p_e = mu[l]
            if h[l] == state:
                p_e = 1 - (A - 1) * mu[l]
            # print("\tis_descenent", u, mutation.node, d, p_e)
            # print(l, u, p_e, h[l], state)
            L[u] = p_t * p_e

        # # compress the likelihoods
        # Up = set()
        # for u in U:
        #     v = tree.parent(u)
        #     while v != tskit.NULL and L[v] == Lambda:
        #         v = tree.parent(v)
        #     if v != tskit.NULL and L[v] == L[u]:
        #         # u has a direct ancestor with the same value, so we can
        #         # compress it out.
        #         L[u] = Lambda
        #     else:
        #         Up.add(u)
        # U = Up

        # # Filter out any internal nodes whose likelihoods don't apply to
        # # any samples.
        # U = set()
        # num_samples = 0
        # N = {u: tree.num_samples(u) for u in Up}
        # for u in sorted(Up, key=lambda u: -tree.time(u)):
        #     v = tree.parent(u)
        #     while v != tskit.NULL and L[v] == Lambda:
        #         v = tree.parent(v)
        #     if v != Lambda:
        #         N[v] -= N[u]
        #     print("Checking for ", u, " n = ", N[u], "v = ", v, ", N[v] = ",
        #             N[v] if v != -1 else "")
        # print("Done:", N)
        # for u, count in N.items():
        #     if count > 0:
        #         U.add(u)
        #     else:
        #         L[u] = Lambda

        # # assert sum(N.values()) == ts.num_samples

        # for u in sorted(Up, key=lambda u: tree.time(u)):
        #     print("Checking ", u, "num_samples= ", num_samples)
        #     if num_samples < ts.num_samples:
        #         U.add(u)
        #         num_samples += tree.num_samples(u)
        #     else:
        #         print("FILTERING", u)
        #         L[u] = Lambda

        # print("AFTER")
        # node_labels = {u: "{}:{:.6g}".format(u, L[u]) for u in U}
        # for u in tree.samples():
        #     if u not in node_labels:
        #         node_labels[u] = "       " # padding
        # print(tree.draw(format="unicode", node_labels=node_labels))

    # for site in tree.sites():
    #     print(site.id, site.mutations[0].node)

    # Traceback
    print(tree.draw(format="unicode"))
    P = np.zeros(m, dtype=int)
    l = m - 1
    u = max(U, key=lambda u: L[u])
    # print("best node = ", u)
    # Get the first sample descendant
    P[l] = min(tree.samples(u))
    # print("initial = ", P[l])
    while l > 0:
        u = P[l]
        P[l - 1] = u
        # print(l, "->", u, T[l])
        while u != tskit.NULL and u not in T[l]:
            u = tree.parent(u)
        if T[l][u]:
            P[l - 1] = min(tree.samples(I[l]))
            # print("RECOMB to ", P[l - 1], "best node = ", I[l])
        l -= 1
    return P




def main():
    np.set_printoptions(linewidth=1000)

    # ts = msprime.simulate(250, recombination_rate=1, mutation_rate=2,
    #         random_seed=2, length=100)
    ts = msprime.simulate(25, recombination_rate=0, mutation_rate=2,
            random_seed=12, length=5.85)
    H = ts.genotype_matrix().T
    print("Shape = ", H.shape)
    h = np.zeros(ts.num_sites, dtype=int)
    h[ts.num_sites // 2:] = 1

    h = H[0].copy()
    h[ts.num_sites // 2:] = H[-1, ts.num_sites // 2:]

    V_tree = np.zeros((ts.num_sites, ts.num_samples))
    V_matrix = np.zeros((ts.num_sites, ts.num_samples))

    r = 1
    # rho = np.zeros(ts.num_sites) + 1
    # mu = np.zeros(ts.num_sites) + 0.1 #1e-30
    np.random.seed(1)
    rho = np.random.random(ts.num_sites)
    mu = np.random.random(ts.num_sites) #* 0.000001
    matrix_path = ls_matrix(h, H, rho, mu, V_matrix)
    path = ls_tree_naive(h, ts, rho, mu, V_tree)
    # path = ls_matrix_vectorised(h, H, rho, mu)
    # assert np.all(path == path2)
    # print("p1", path)
    # print("p2", path2)
    match = H[path, np.arange(ts.num_sites)]
    print("h     = ", h)
    print("path  = ", path)
    print("pathm = ", matrix_path)
    print("match = ", match)
    print("eq    = ", np.all(h == match))
    print("patheq= ", np.all(path == matrix_path))

    print("TREE")
    print(V_tree)
    print("MATRIX")
    print(V_matrix)
    print("all close?", np.allclose(V_tree, V_matrix))
    print((V_tree != V_matrix).astype(int))



if __name__ == "__main__":
    main()
