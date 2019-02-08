"""
Direct implementations of the algorithms in the paper.
"""
import itertools
import concurrent.futures
import math
import colorsys

import msprime
import tskit
import numpy as np
import cairo

class CairoVisualiser(object):

    def __init__(self, h, H, rho, mu, width=800, height=600):
        self.h = h
        self.H = H
        self.rho = rho
        self.mu = mu

        self.path, state = ls_matrix(h, H, rho, mu, return_internal=True)
        self.I = state["I"]
        self.V = state["V"]
        self.T = state["T"]

        self.width = width
        self.height = height

    def centre_text(self, cr, x, y, text):
        xbearing, ybearing, width, height, xadvance, yadvance = cr.text_extents(text)
        cr.move_to(x + 0.5 - xbearing - width / 2, y + 0.5 - ybearing - height / 2)
        cr.show_text(text)

    def draw(self, output_file):
        n, m = self.H.shape
        w = self.width
        h = self.height
        f_w = 0.9
        f_h = 0.7
        cell_w = w * f_w / m
        cell_h = h * f_h / n
        matrix_origin = (w - w * f_w) / 2, (h - h * f_h) / 2

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        cr = cairo.Context(surface)
        # Transform to normal cartesian coordinate system
        # TODO be nice to use this, but it inverts the text as well.
        # cr.transform(cairo.Matrix(yy=-1, y0=h))

        # Set a background color
        cr.save()
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()
        cr.restore()

        for j in range(n):
            y = matrix_origin[1] + cell_h * j
            for k in range(m):
                x = matrix_origin[0] + cell_w * k
                hsv = (0.5, 0.85, self.V[k, j])
                cr.set_source_rgb(*colorsys.hsv_to_rgb(*hsv))
                cr.rectangle(x, y, cell_w, cell_h)
                cr.fill()

        # Fill in the text.
        cr.select_font_face(
            "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(10)
        cr.set_source_rgb(0.5, 0.5, 0.5)
        for j in range(n):
            y = matrix_origin[1] + cell_h * j + cell_h / 2
            x = matrix_origin[0] - cell_w + cell_w / 2
            self.centre_text(cr, x, y, str(j))
            for k in range(m):
                x = matrix_origin[0] + cell_w * k + cell_w / 2
                self.centre_text(cr, x, y, str(self.H[j, k]))

        # Draw the grid
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.set_line_width(2)
        for j in range(n + 1):
            y = matrix_origin[1] + cell_h * j
            cr.move_to(matrix_origin[0], y)
            x = matrix_origin[0] + cell_w * m
            cr.line_to(x, y)
        cr.stroke()

        for k in range(m + 1):
            x = matrix_origin[0] + cell_w * k
            cr.move_to(x, matrix_origin[1])
            y = matrix_origin[1] + cell_h * n
            cr.line_to(x, y)
        cr.stroke()

        # Draw the query string
        for k in range(m + 1):
            x = matrix_origin[0] + cell_w * k
            cr.move_to(x, matrix_origin[1] - cell_h)
            y = matrix_origin[1] - 2 * cell_h
            cr.line_to(x, y)
            if k < m:
                x = matrix_origin[0] + cell_w * k + cell_w / 2
                y = matrix_origin[1] - 1.5 * cell_h
                self.centre_text(cr, x, y, str(self.h[k]))

        for y in [matrix_origin[1] - cell_h, matrix_origin[1] - 2 * cell_h]:
            cr.move_to(matrix_origin[0], y)
            x = matrix_origin[0] + cell_w * m
            cr.line_to(x, y)
        cr.stroke()

        # Draw the path
        cr.set_line_width(2)
        cr.set_source_rgb(1, 0, 0)
        for k in range(m):
            j = self.path[j]
            y = matrix_origin[1] + cell_h * j
            x = matrix_origin[0] + cell_w * k
            cr.rectangle(x, y, cell_w, cell_h)
            cr.stroke()

        surface.write_to_png(output_file)



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


def in_sorted(values, j):
    # Take advantage of the fact that the numpy array is sorted.
    ret = False
    index = np.searchsorted(values, j)
    if index < values.shape[0]:
        ret = values[index] == j
    return ret



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
    V = np.ones(n)
    T = np.zeros((m, n), dtype=int)

    for l in range(m):
        for j in range(n):
            for k in range(n):
                p_t = (1 - rho[l] - rho[l] / n) * V[j]
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
        # Normalise
        I[l] = np.argmax(V)
        V /= V[I[l]]

    # Traceback
    P = np.zeros(m, dtype=int)
    l = m - 1
    P[l] = I[l]
    while l > 0:
        j = P[l]
        if in_sorted(T[l], j):
            j = I[l - 1]
        P[l - 1] = j
        l -= 1
    return P


def ls_matrix(h, H, rho, mu, return_internal=False):
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

    if return_internal:
        V_matrix = np.zeros((m, n))

    for l in range(m):
        if return_internal:
            V_matrix[l] = V
        p_neq = rho[l] / n
        for j in range(n):
            # NOTE Question here: Should we take into account what the best
            # haplotype at the previous site actually was and avoid
            # false recombinations? Does it matter?
            p_t = (1 - rho[l] - rho[l] / n) * V[j]
            if p_neq > p_t:
                p_t = p_neq
                T[l].add(j)
            p_e = mu[l]
            if H[j, l] == h[l]:
                p_e = 1 - (A - 1) * mu[l]
            V[j] = p_t * p_e
        I[l] = np.argmax(V)
        V /= V[I[l]]

    # Traceback
    P = np.zeros(m, dtype=int)
    l = m - 1
    P[l] = I[l]
    j = P[l]
    while l > 0:
        if j in T[l]:
            j = I[l - 1]
        P[l - 1] = j
        l -= 1

    if return_internal:
        return P, {"I": I, "V": V_matrix, "T": T}
    else:
        return P


def ls_tree_naive(h, ts, rho, mu, return_internal=False):
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
    L[tree.root] = 1
    U = {tree.root}

    if return_internal:
        V_matrix = np.zeros((m, n))

    for site in tree.sites():
        assert len(U) > 0
        assert len(site.mutations) == 1
        mutation = site.mutations[0]

        if return_internal:
            V = np.zeros(n)
            for j in range(n):
                v = j
                while L[v] == Lambda:
                    v = tree.parent(v)
                assert v != tskit.NULL
                V[j] = L[v]
            V_matrix[site.id] = V
        # print(site.id, h[site.id], V)

        l = site.id

        u = mutation.node
        while u != tskit.NULL and L[u] == Lambda:
            u = tree.parent(u)
        # If there is no likelihood in this part of the tree we assign likelihood
        # of zero to the mutation node.
        L[mutation.node] = 0 if u == tskit.NULL else L[u]
        U.add(mutation.node)

        p_neq = rho[l] / n
        for u in U:
            p_t = (1 - rho[l] - rho[l] / n) * L[u]
            # print(l, u, p_t, p_neq, p_neq > p_t)
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

        # compress the likelihoods
        Up = set()
        for u in U:
            v = tree.parent(u)
            while v != tskit.NULL and L[v] == Lambda:
                v = tree.parent(v)
            if v != tskit.NULL and L[v] == L[u]:
                # u has a direct ancestor with the same value, so we can
                # compress it out.
                L[u] = Lambda
            else:
                Up.add(u)
        # Filter out any internal nodes whose likelihoods don't apply to
        # any samples.
        U = set()
        num_samples = 0
        N = {u: tree.num_samples(u) for u in Up}
        for u in sorted(Up, key=lambda u: -tree.time(u)):
            v = tree.parent(u)
            while v != tskit.NULL and L[v] == Lambda:
                v = tree.parent(v)
            if v != Lambda:
                N[v] -= N[u]
        for u, count in N.items():
            if count > 0:
                U.add(u)
            else:
                L[u] = Lambda
        assert sum(N.values()) == ts.num_samples

        # Find max likelihood and normalise
        I[l] = max(U, key=lambda u: (L[u], -u))
        max_L = L[I[l]]
        for u in U:
            L[u] /= max_L
        print(l, {u: L[u] for u in U})

    # for l in range(m):
    #     print(l, T[l])
    # Traceback
    print(tree.draw(format="unicode"))

    P = np.zeros(m, dtype=int)
    l = m - 1
    print("best node = ", I[l])
    # Get the first sample descendant
    P[l] = min(tree.samples(I[l]))
    print("initial = ", P[l])
    while l > 0:
        u = P[l]
        P[l - 1] = u
        # print(l, "->", u, T[l])
        while u != tskit.NULL and u not in T[l]:
            u = tree.parent(u)
        if T[l][u]:
            P[l - 1] = min(tree.samples(I[l - 1]))
            print("RECOMB at", l, "to ", P[l - 1], "best node = ", I[l - 1])
        l -= 1

    if return_internal:
        return P, {"I": I, "V": V_matrix, "T": T}
    else:
        return P


def verify_tree_algorithm(ts):

    H = ts.genotype_matrix().T
    haplotypes = []
    h = np.zeros(ts.num_sites, dtype=int)
    h[ts.num_sites // 2:] = 1
    haplotypes.append(h)
    h = H[0].copy()
    h[ts.num_sites // 2:] = H[-1, ts.num_sites // 2:]
    haplotypes.append(h)
    haplotypes.append(H[-1][:])
    haplotypes.append(np.zeros(ts.num_sites, dtype=np.int8))
    haplotypes.append(np.ones(ts.num_sites, dtype=np.int8))
    for _ in range(10):
        h = np.random.randint(0, 2, ts.num_sites)
        haplotypes.append(h)

    rhos = [
        np.zeros(ts.num_sites) + 1,
        np.zeros(ts.num_sites) + 1e-20,
        np.random.random(ts.num_sites)]
    mus = [
        np.zeros(ts.num_sites) + 1,
        np.zeros(ts.num_sites) + 1e-20,
        np.random.random(ts.num_sites)]

    for h, mu, rho in itertools.product(haplotypes, mus, rhos):
        matrix_path, m_state = ls_matrix(h, H, rho, mu, V_matrix, return_internal=True)
        tree_path, t_state = ls_tree_naive(h, ts, rho, mu, return_internal=True)
        V_tree = t_state["V"]
        V_matrix = m_state["V"]
        # assert np.all(matrix_path == tree_path)
        assert np.allclose(V_tree, V_matrix)


def verify_worker(work):
    n, length = work
    ts = msprime.simulate(n, recombination_rate=0, mutation_rate=2,
            random_seed=12, length=length)
    verify_tree_algorithm(ts)
    return ts.num_samples, ts.num_sites


def verify():
    work = itertools.product([3, 5, 20, 50], [1, 10, 100, 500])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_work = {executor.submit(verify_worker, w): w for w in work}
        for future in concurrent.futures.as_completed(future_to_work):
            n, m = future.result()
            print("Verify n =", n, "num_sites =", m)


def develop():
    # ts = msprime.simulate(250, recombination_rate=1, mutation_rate=2,
    #         random_seed=2, length=100)
    ts = msprime.simulate(15, recombination_rate=0, mutation_rate=2,
            random_seed=12, length=1.85)

    H = ts.genotype_matrix().T
    print("Shape = ", H.shape)
    h = np.zeros(ts.num_sites, dtype=int)
    h[ts.num_sites // 2:] = 1

    h = H[0].copy()
    h[ts.num_sites // 2:] = H[-1, ts.num_sites // 2:]

    r = 1
    # rho = np.zeros(ts.num_sites) + 1
    # mu = np.zeros(ts.num_sites) + 0.01 #1e-30
    np.random.seed(1)
    rho = np.random.random(ts.num_sites)
    mu = np.random.random(ts.num_sites) * 0.000001
    # matrix_path, matrix_state = ls_matrix(h, H, rho, mu, return_internal=True)
    # print(H)

    viz = CairoVisualiser(h, H, rho, mu)
    viz.draw("output.png")



    # path, tree_state = ls_tree_naive(h, ts, rho, mu, return_internal=True)
    # # path = ls_matrix_vectorised(h, H, rho, mu)
    # # assert np.all(path == path2)
    # # print("p1", path)
    # # print("p2", path2)
    # match = H[path, np.arange(ts.num_sites)]
    # # print("h     = ", h)
    # print("path  = ", path)
    # print("pathm = ", matrix_path)
    # # print("match = ", match)
    # # print("eq    = ", np.all(h == match))
    # print("patheq= ", np.all(path == matrix_path))
    # print(np.where(path != matrix_path))

    # print("all close?", np.allclose(matrix_state["V"], tree_state["V"]))


def main():
    np.set_printoptions(linewidth=1000)

    # verify()
    develop()


if __name__ == "__main__":
    main()
