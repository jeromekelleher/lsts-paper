"""
Direct implementations of the algorithms in the paper.
"""
import itertools
import concurrent.futures
import math
import colorsys
import statistics
import heapq
import collections
import string
import random

import matplotlib.pyplot as plt

import msprime
import tskit
import numpy as np
import cairo

import hmm


def fitch_sets(tree, labels):
    """
    Return the list of Fitch sets for the nodes in this tree with the
    specified set of sample labels.
    """
    A = [set() for _ in range(tree.num_nodes)]
    for label, sample in zip(labels, tree.tree_sequence.samples()):
        A[sample] = {label}
    for u in tree.nodes(order="postorder"):
        if len(A[u]) == 0:
            A[u] = set.intersection(*[A[v] for v in tree.children(u)])
            if len(A[u]) == 0:
                A[u] = set.union(*[A[v] for v in tree.children(u)])
    return A

def incremental_fitch_sets(ts, labels):
    """
    Returns an iterator over the Fitch sets for the specified tree sequence.
    """
    K = np.max(labels) + 1
    # Quintuply linked tree.
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    left_sib = np.zeros(ts.num_nodes, dtype=int) - 1
    right_sib = np.zeros(ts.num_nodes, dtype=int) - 1
    left_child = np.zeros(ts.num_nodes, dtype=int) - 1
    right_child = np.zeros(ts.num_nodes, dtype=int) - 1
    # Fitch sets.
    A = [set() for _ in range(ts.num_nodes)]
    for label, sample in zip(labels, ts.samples()):
        A[sample] = {label}

    def remove_edge(edge):
        p = edge.parent
        c = edge.child
        lsib = left_sib[c]
        rsib = right_sib[c]
        if lsib == -1:
            left_child[p] = rsib
        else:
            right_sib[lsib] = rsib
        if rsib == -1:
            right_child[p] = lsib
        else:
            left_sib[rsib] = lsib
        parent[c] = -1
        left_sib[c] = -1
        right_sib[c] = -1

    def insert_edge(edge):
        p = edge.parent
        c = edge.child
        assert parent[c] == -1, "contradictory edges"
        parent[c] = p
        u = right_child[p]
        if u == -1:
            left_child[p] = c
            left_sib[c] = -1
            right_sib[c] = -1
        else:
            right_sib[u] = c
            left_sib[c] = u
            right_sib[c] = -1
        right_child[p] = c

    def propagate_fitch(u):
        """
        Starting from node u, propagate Fitch sets changes up the tree.
        """
        while u != -1:
            children = []
            v = left_child[u]
            while v != -1:
                children.append(v)
                v = right_sib[v]
            if len(children) == 0:
                A[u] = set()
            else:
                a = set.intersection(*[A[v] for v in children])
                if len(a) == 0:
                    a = set.union(*[A[v] for v in children])
                # If there's no change at this node then we can't have any changes
                # further up the tree.
                if A[u] == a:
                    break
                A[u] = a
            u = parent[u]

    for (left, right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            remove_edge(edge)
            propagate_fitch(edge.parent)

        for edge in edges_in:
            insert_edge(edge)
            propagate_fitch(edge.parent)
        yield A



def count_parismony(tree, labels):
    ts = tree.tree_sequence
    count = np.zeros((ts.num_nodes, np.max(labels) + 1), dtype=int)
    for j, sample in enumerate(ts.samples()):
        count[sample, labels[j]] = 1

    for u in tree.nodes(order="postorder"):
        zeros = np.ones(count.shape[1], dtype=int)
        for v in tree.children(u):
            count[u] += count[v]
            zeros = np.logical_and(zeros, count[v] > 0)
        if np.sum(zeros) > 0:
            count[u] *= zeros

    ancestral_state = np.where(count[tree.root] > 0)[0][0]
    nodes = []
    states = []
    stack = [(tree.root, ancestral_state)]
    while len(stack) > 0:
        u, state = stack.pop()
        for v in tree.children(u):
            child_state = state
            if count[v, state] == 0:
                child_state = np.where(count[v] > 0)[0][0]
                nodes.append(v)
                states.append(child_state)
            stack.append((v, child_state))
    return ancestral_state, np.array(nodes), np.array(states), count


def np_fitch_counts(tree, labels):
    """
    Return the Fitch sets in which we count the number of immediate children
    that are in the Fitch set.
    """
    ts = tree.tree_sequence
    K = np.max(labels) + 1
    A = np.zeros((ts.num_nodes, K), dtype=int)
    for label, sample in zip(labels, tree.tree_sequence.samples()):
        A[sample, label] = 1
    for u in tree.nodes(order="postorder"):
        if tree.is_internal(u):
            U = np.zeros(K, dtype=int)
            I = np.zeros(K, dtype=int)
            mask = np.ones(K, dtype=int)
            for v in tree.children(u):
                U += A[v] > 0
                mask = np.logical_and(mask, A[v] > 0)
                I[np.logical_not(mask)] = 0
                I[mask] += A[v][mask] > 0
            A[u] = I
            if np.sum(mask) == 0:
                A[u] = U
    return A


def incremental_fitch_counts(ts, labels):
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    K = np.max(labels) + 1
    A = np.zeros((ts.num_nodes, K), dtype=int)
    I = np.zeros((ts.num_nodes, K), dtype=int)
    U = np.zeros((ts.num_nodes, K), dtype=int)
    M = np.ones((ts.num_nodes, K), dtype=int)

    for label, sample in zip(labels, ts.samples()):
        A[sample, label] = 1

    for (left, right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            parent[edge.child] = -1

        for edge in edges_in:
            parent[edge.child] = edge.parent
            u = edge.parent
            v = edge.child
            while u != -1:
                U[u] += A[v] > 0
                M[u] = np.logical_and(M[u], A[v] > 0)
                I[u][np.logical_not(M[u])] = 0
                I[u][M[u]] += A[v][M[u]] > 0
                A[u] = I[u]
                if np.sum(M[u]) == 0:
                    A[u] = U[u]
                v = u
                u = parent[u]
        for j in range(ts.num_nodes):
            print(j, M[j], I[j], U[j])
        yield A

def incremental_fitch_dev():
    ts = msprime.simulate(7, recombination_rate=15, random_seed=2)

    labels = np.zeros(ts.num_samples, dtype=np.uint8)
    labels[ts.sample_size // 3:] = 1
    labels[2 * ts.sample_size // 3:] = 2
    print(labels)

    for tree, A3 in zip(ts.trees(), incremental_fitch_counts(ts, labels)):

        # node_labels = {u: "{}:{}".format(u, A2[u]) for u in tree.nodes()}
        # t2 = tree.draw(format="unicode", node_labels=node_labels)
        # for l1, l2 in zip(t1.splitlines(), t2.splitlines()):
        #     print(l1, "|", l2)

        # ancestral_state1, (nodes1, _, states1) = tree.reconstruct(labels)
        # ancestral_state2, nodes2, states2, count = count_parismony(tree, labels)
        # assert ancestral_state1 == ancestral_state2
        # assert np.all(nodes1 == nodes2)
        # assert np.all(states1 == states2)
        # for u in tree.nodes():
        #     end = "" if np.all(A2[u] == count[u]) else "*"
        #     print(u, A2[u], count[u], end, sep="\t")
        # # print(A2)
        # # print(count)

        A1 = fitch_sets(tree, labels)
        A2 = np_fitch_counts(tree, labels)
        node_labels = {u: "{}:{}:{}".format(u, A1[u], A2[u]) for u in tree.nodes()}
        t1 = tree.draw(format="unicode", node_labels=node_labels)
        print(t1)
        # print(A2)
        # print(A3)
        print(np.all(A2 == A3))
        for j in range(ts.num_nodes):
            # print(fs, counts)
            if not np.all(A2[j] == A3[j]):
                print(j, A2[j], A3[j])
            assert set(np.where(A2[j] > 0)[0]) == A1[j]
        # assert A1 == A2
        # print("ancestral_state1 = ", ancestral_state, node, state)
        # print("ancestral_state2 = ", ancestral_state2, nodes2, states2)
        # assert len(nodes2) == len(node)

#         N = {u: tree.num_samples(u) for u in f}
#         for u in sorted(f.keys(), key=lambda u: -tree.time(u)):
#             v = tree.parent(u)
#             while v != tskit.NULL and v not in f:
#                 v = tree.parent(v)
#             if v != tskit.NULL:
#                 N[v] -= N[u]


def generate_site_mutations(tree, position, mu, site_table, mutation_table,
                            multiple_per_node=True):
    """
    Generates mutations for the site at the specified position on the specified
    tree. Mutations happen at rate mu along each branch. The site and mutation
    information are recorded in the specified tables.  Note that this records
    more than one mutation per edge.
    """
    assert tree.interval[0] <= position < tree.interval[1]
    states = {"A", "C", "G", "T"}
    state = random.choice(sorted(list(states)))
    site_table.add_row(position, state)
    site = site_table.num_rows - 1
    for root in tree.roots:
        stack = [(root, state, tskit.NULL)]
        while len(stack) != 0:
            u, state, parent = stack.pop()
            if u != root:
                branch_length = tree.branch_length(u)
                x = random.expovariate(mu)
                new_state = state
                while x < branch_length:
                    new_state = random.choice(sorted(list(states - set(state))))
                    if multiple_per_node and (state != new_state):
                        mutation_table.add_row(site, u, new_state, parent)
                        parent = mutation_table.num_rows - 1
                        state = new_state
                    x += random.expovariate(mu)
                else:
                    if (not multiple_per_node) and (state != new_state):
                        mutation_table.add_row(site, u, new_state, parent)
                        parent = mutation_table.num_rows - 1
                        state = new_state
            stack.extend(reversed([(v, state, parent) for v in tree.children(u)]))


def jukes_cantor(ts, num_sites, mu, multiple_per_node=True, seed=None):
    """
    Returns a copy of the specified tree sequence with Jukes-Cantor mutations
    applied at the specfied rate at the specifed number of sites. Site positions
    are chosen uniformly.
    """
    random.seed(seed)
    positions = [ts.sequence_length * random.random() for _ in range(num_sites)]
    positions.sort()
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    t = ts.first()
    for position in positions:
        while position >= t.interval[1]:
            t.next()
        generate_site_mutations(
            t, position, mu, tables.sites, tables.mutations,
            multiple_per_node=multiple_per_node)
    return tables.tree_sequence()


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
        self.match = self.H[self.path, np.arange(H.shape[1])]

        self.width = width
        self.height = height

    def __str__(self):
        print("path = ", self.path)
        print("match = ", self.match)
        print("h     = ", self.h)

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
        cr.set_font_size(12)
        cr.set_source_rgb(0.5, 0.5, 0.5)
        label_x = matrix_origin[0] - cell_w + cell_w / 2
        for j in range(n):
            y = matrix_origin[1] + cell_h * j + cell_h / 2
            self.centre_text(cr, label_x, y, str(j))
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

        # Draw the query string mismatches
        cr.set_source_rgb(0.5, 0.5, 0.5)
        for k in range(m):
            if self.match[k] != self.h[k]:
                x = matrix_origin[0] + cell_w * k
                y = matrix_origin[1] - 2 * cell_h
                cr.rectangle(x, y, cell_w, cell_h)
                cr.fill()

        # Draw the query string
        cr.set_source_rgb(0, 0, 0)
        for k in range(m):
            x = matrix_origin[0] + cell_w * k + cell_w / 2
            y = matrix_origin[1] - 1.5 * cell_h
            self.centre_text(cr, x, y, str(self.h[k]))

        for k in range(m + 1):
            x = matrix_origin[0] + cell_w * k
            cr.move_to(x, matrix_origin[1] - cell_h)
            y = matrix_origin[1] - 2 * cell_h
            cr.line_to(x, y)

        for y in [matrix_origin[1] - cell_h, matrix_origin[1] - 2 * cell_h]:
            cr.move_to(matrix_origin[0], y)
            x = matrix_origin[0] + cell_w * m
            cr.line_to(x, y)
        cr.stroke()

        # Draw the path
        cr.set_line_width(2)
        cr.set_source_rgb(1, 0, 0)
        for k in range(m):
            j = self.path[k]
            y = matrix_origin[1] + cell_h * j + cell_h / 2
            x = matrix_origin[0] + cell_w * k
            cr.rectangle(x, y, cell_w, 1)
            cr.stroke()

        # Draw the highest proba cells.
        cr.set_line_width(3)
        cr.set_source_rgb(0, 0, 1)
        for k in range(m):
            j = self.I[k]
            y = matrix_origin[1] + cell_h * j
            x = matrix_origin[0] + cell_w * k
            cr.rectangle(x, y, cell_w, cell_h)
            cr.stroke()

        # Draw the recombination needed cells
        cr.set_line_width(2)
        cr.set_source_rgb(0, 1, 0)
        for k in range(m):
            for j in self.T[k]:
                y = matrix_origin[1] + cell_h * j
                x = matrix_origin[0] + cell_w * k
                cr.rectangle(x, y, cell_w, cell_h)
                cr.stroke()

        # Draw the mutation and recombinations weights
        cr.set_line_width(2)
        rho_y = matrix_origin[1] + cell_h * (n + 0.5)
        mu_y = matrix_origin[1] + cell_h * (n + 1.5)
        for k in range(m):
            x = matrix_origin[0] + k * cell_w
            v = self.rho[k]
            cr.rectangle(x, rho_y, cell_w, cell_h)
            cr.set_source_rgb(0, 0, 0)
            cr.stroke_preserve()
            cr.set_source_rgb(v, v, v)
            cr.fill()

            v = self.mu[k]
            cr.rectangle(x, mu_y, cell_w, cell_h)
            cr.set_source_rgb(0, 0, 0)
            cr.stroke_preserve()
            cr.set_source_rgb(v, v, v)
            cr.fill()

        cr.set_source_rgb(0, 0, 0)
        self.centre_text(cr, label_x, mu_y + cell_h / 2, "μ")
        self.centre_text(cr, label_x, rho_y + cell_h / 2, "ρ")

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
        p_t = (1 - rho[l] + rho[l] / n) * V
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


def ls_forward_matrix_unscaled(h, H, rho, mu):
    """
    Simple matrix based method for LS forward algorithm.
    """
    assert rho[0] == 0
    n, m = H.shape
    A = 2 # Fixing to binary for now.
    F = np.zeros((n, m))
    f = np.zeros(n) + 1 / n

    for l in range(0, m):
        s = np.sum(f)
        for j in range(n):
            p_t = f[j] * (1 - rho[l]) + s * rho[l] / n
            p_e = mu[l]
            if H[j, l] == h[l]:
                p_e = 1 - (A - 1) * mu[l]
            f[j] = p_t * p_e
        F[:, l] = f
    return F


def ls_forward_matrix(h, alleles, H, rho, mu):
    """
    Simple matrix based method for LS forward algorithm.
    """
    assert rho[0] == 0
    n, m = H.shape
    F = np.zeros((n, m))
    S = np.zeros(m)
    f = np.zeros(n) + 1 / n

    for l in range(0, m):
        for j in range(n):
            p_t = f[j] * (1 - rho[l]) + rho[l] / n
            p_e = mu[l]
            if H[j, l] == h[l]:
                p_e = 1 - (len(alleles[l]) - 1) * mu[l]
            f[j] = p_t * p_e
        S[l] = np.sum(f)
        f /= S[l]
        F[:, l] = f
    return F, S


def fitch(tree, f):
    """
    Use full parsimony method from tskit.
    """
    genotypes = np.zeros(tree.tree_sequence.num_samples, dtype=np.uint8)
    alleles = list(set(f.values()))
    for j, u in enumerate(tree.tree_sequence.samples()):
        v = u
        while v not in f:
            v = tree.parent(v)
        genotypes[j] = alleles.index(f[v])

    ancestral_state, (node, _, state) = tree.reconstruct(genotypes)
    f = {tree.root: alleles[ancestral_state]}
    for u, s in zip(node, state):
        f[u] = alleles[s]

    return f

def draw_tree(tree, f):
    N = {u: tree.num_samples(u) for u in f}
    for u in sorted(f.keys(), key=lambda u: -tree.time(u)):
        v = tree.parent(u)
        while v != tskit.NULL and v not in f:
            v = tree.parent(v)
        if v != tskit.NULL:
            N[v] -= N[u]

    frequency = collections.Counter()
    for node, value in f.items():
        frequency[value] += N[node]
    label = {
        v[0]: string.ascii_letters[j].upper()
        for j, v in enumerate(frequency.most_common())}
    node_labels = {u: "{}:{}".format(label[f[u]], N[u]) for u in f}
    return tree.draw(format="unicode", node_labels=node_labels)


def compress(tree, f):
    # Quantise f
    f = {u: round(f[u], 10) for u in f}

    before = draw_tree(tree, f)
    f = fitch(tree, f)
    after = draw_tree(tree, f)
    for l1, l2 in zip(before.splitlines(), after.splitlines()):
        print(l1, "|", l2)

    return f

def get_state(tree, site, alleles, u):
    # this should also work for multiple mutations, since mutations are listed
    # in order going down the tree.
    mutations = {mut.node: mut.derived_state for mut in site.mutations}
    while u != tskit.NULL and u not in mutations:
        u = tree.parent(u)
    allele = mutations.get(u, site.ancestral_state)
    return alleles[site.id].index(allele)


def ls_forward_tree_naive(h, alleles, ts, rho, mu):
    """
    Forward matrix computation based on a single tree.
    """
    n, m = ts.num_samples, ts.num_sites
    F = [None for _ in range(m)]
    S = np.zeros(m)
    f = {u: 1 / n  for u in ts.samples()}

    for tree in ts.trees():
        for site in tree.sites():
            l = site.id
            # print("l = ", l, h[l])
            for mutation in site.mutations:
                u = mutation.node
                while u != tskit.NULL and u not in f:
                    u = tree.parent(u)
                f[mutation.node] = 0 if u == tskit.NULL else f[u]
            for u in f.keys():
                p_t = f[u] * (1 - rho[l]) + rho[l] / n
                state = get_state(tree, site, alleles, u)
                p_e = mu[l]
                if h[l] == state:
                    p_e = 1 - (len(alleles[l]) - 1) * mu[l]
                f[u] = p_t * p_e

            f = compress(tree, f)

            N = {u: tree.num_samples(u) for u in f}
            for u in sorted(f.keys(), key=lambda u: -tree.time(u)):
                v = tree.parent(u)
                while v != tskit.NULL and v not in f:
                    v = tree.parent(v)
                if v != tskit.NULL:
                    N[v] -= N[u]

            S[l] = sum(N[u] * f[u] for u in f)
            f = {u: f[u] / S[l] for u in f}
            F[l] = f.copy()

        # Scatter f out again for the next tree
        fp = {}
        for u in tree.samples():
            v = u
            while v not in f:
                v = tree.parent(v)
            fp[u] = f[v]
        f = fp
    # print(V)

    # print(tree.draw(format="unicode", node_labels={u: str(N[u]) for u in f}))

    return F, S

def ls_forward_tree(h, alleles, ts, rho, mu):
    """
    Forward matrix computation based on a tree sequence.
    """
    n, m = ts.num_samples, ts.num_sites
    F = [None for _ in range(m)]
    S = np.zeros(m)
    f = {}
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    A = [set() for _ in range(ts.num_nodes)]

    for u in ts.samples():
        f[u] = 1 / n
        A[u] = {f[u]}

    tree = tskit.Tree(ts)
    for (left, right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            parent[edge.child] = -1
            # v = edge.parent
            # while v != -1:
            #     A[v] -= A[edge.child]
            #     v = parent[v]
        for edge in edges_in:
            parent[edge.child] = edge.parent
            # if not A[edge.child] <= A[edge.parent]:
            #     A[edge.parent] |= A[edge.child]
        tree.next()
        for site in tree.sites():
            l = site.id
            # print("l = ", l, h[l])
            for mutation in site.mutations:
                u = mutation.node
                while u != tskit.NULL and u not in f:
                    u = parent[u]
                f[mutation.node] = 0 if u == tskit.NULL else f[u]

            mutations = {mut.node: mut.derived_state for mut in site.mutations}
            for u in f.keys():
                # Get the state at u. TODO we can add a state_cache here.
                v = u
                while v != tskit.NULL and v not in mutations:
                    v = parent[v]
                allele = mutations.get(v, site.ancestral_state)
                state = alleles[site.id].index(allele)

                # Compute the F value for u.
                p_t = f[u] * (1 - rho[l]) + rho[l] / n
                p_e = mu[l]
                if h[l] == state:
                    p_e = 1 - (len(alleles[l]) - 1) * mu[l]
                f[u] = round(p_t * p_e, 8)

            # f = compress(tree, f)

            N = {u: tree.num_samples(u) for u in f}
            for u in sorted(f.keys(), key=lambda u: -tree.time(u)):
                v = parent[u]
                while v != tskit.NULL and v not in f:
                    v = parent[v]
                if v != tskit.NULL:
                    N[v] -= N[u]

            S[l] = sum(N[u] * f[u] for u in f)
            f = {u: f[u] / S[l] for u in f}
            F[l] = f.copy()

        # Scatter f out again for the next tree
        A = [set() for _ in range(ts.num_nodes)]
        fp = {}
        for u in tree.samples():
            v = u
            while v not in f:
                v = tree.parent(v)
            fp[u] = f[v]
            A[u] = {f[v]}
        f = fp
        for u in tree.nodes(order="postorder"):
            if tree.is_internal(u):
                A[u] = set.intersection(*[A[v] for v in tree.children(u)])
                if len(A[u]) == 0:
                    A[u] = set.union(*[A[v] for v in tree.children(u)])

        val_map = {
                v: string.ascii_letters[j].upper()
                for j, v in enumerate(set(f.values()))}
        node_labels = {u: str({val_map[x] for x in A[u]}) for u in tree.nodes()}
        print(tree.draw(format="unicode", node_labels=node_labels, width=400))
    # print(V)

    # print(tree.draw(format="unicode", node_labels={u: str(N[u]) for u in f}))

    return F, S



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
            p_t = (1 - rho[l] + rho[l] / n) * V[j]
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
            p_t = (1 - rho[l] + rho[l] / n) * L[u]
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
        # print(l, {u: L[u] for u in U})

    # for l in range(m):
    #     print(l, T[l])
    # Traceback
    # print(tree.draw(format="unicode"))

    P = np.zeros(m, dtype=int)
    l = m - 1
    # print("best node = ", I[l])
    # Get the first sample descendant
    P[l] = min(tree.samples(I[l]))
    # print("initial = ", P[l])
    while l > 0:
        u = P[l]
        P[l - 1] = u
        # print(l, "->", u, T[l])
        while u != tskit.NULL and u not in T[l]:
            u = tree.parent(u)
        if T[l][u]:
            P[l - 1] = min(tree.samples(I[l - 1]))
            # print("RECOMB at", l, "to ", P[l - 1], "best node = ", I[l - 1])
        l -= 1

    if return_internal:
        return P, {"I": I, "V": V_matrix, "T": T}
    else:
        return P


def ls_hmm(H, rho, mu):

    n, m = H.shape
    states = np.arange(n)
    start_prob = {j: 1 / n for j in range(n)}

    def trans_prob(state1, state2, site):
        ret = rho[site] / n
        if state1 == state2:
            ret = 1 - rho[site] + rho[site] / n
        return ret

    def emit_prob(state, symbol, site):
        ret = mu[site]
        if H[state, site] == symbol:
            ret = 1 - mu[site]
        return ret

    model = hmm.Model(states, [0, 1], start_prob, trans_prob, emit_prob)
    return model

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

    # print(tree.draw(format="unicode"))
    for h, mu, rho in itertools.product(haplotypes, mus, rhos):
        matrix_path, m_state = ls_matrix(h, H, rho, mu, return_internal=True)
        tree_path, t_state = ls_tree_naive(h, ts, rho, mu, return_internal=True)
        V_tree = t_state["V"]
        V_matrix = m_state["V"]
        # assert np.all(matrix_path == tree_path)
        assert np.allclose(V_tree, V_matrix)
        for tree in ts.trees():
            for site in tree.sites():
                T_tree = t_state["T"][site.id]
                T_matrix = m_state["T"][site.id]
                S = set()
                for u in ts.samples():
                    v = u
                    while v not in T_tree:
                        v = tree.parent(v)
                    if T_tree[v]:
                        S.add(u)
                assert S == T_matrix

                I_tree = t_state["I"][site.id]
                I_matrix = m_state["I"][site.id]
                V_site = V_tree[site.id]
                Vi = V_site[list(tree.samples(I_tree))]
                print(I_tree, I_matrix)
                print(Vi, V_site[I_matrix])

                # print(V_tree[site.id][I_tree],
                # print(I_matrix)



def verify_worker(work):
    n, length = work
    ts = msprime.simulate(n, recombination_rate=0, mutation_rate=2,
            random_seed=12, length=length)
    verify_tree_algorithm(ts)
    return ts.num_samples, ts.num_sites


def verify():
    work = itertools.product([3, 5, 20, 50], [1, 10, 100, 500])
    for w in work:
        verify_worker(w)
        print("Verify ", w)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     future_to_work = {executor.submit(verify_worker, w): w for w in work}
    #     for future in concurrent.futures.as_completed(future_to_work):
    #         n, m = future.result()
    #         print("Verify n =", n, "num_sites =", m)

def decode_ts_matrix(ts, F_tree):
    """
    Decodes the specified tree encoding of the probabilities into an explicit
    matrix.
    """
    F = np.zeros((ts.num_samples, ts.num_sites))
    assert len(F_tree) == ts.num_sites
    for tree in ts.trees():
        for site in tree.sites():
            f = F_tree[site.id]
            for j, u in enumerate(ts.samples()):
                while u not in f:
                    u = tree.parent(u)
                F[j, site.id] = f[u]
    return F

def plot_encoding_efficiency():
    for n in [10, 100, 1000, 10000, 10**5]:
        for L in [1, 10, 100]:
            ts = msprime.simulate(
                n, recombination_rate=1, mutation_rate=2, random_seed=13, length=L)
            ts = jukes_cantor(ts, L * 10, 0.05, seed=1, multiple_per_node=False)

            # rho = np.zeros(ts.num_sites) + 0.1
            # mu = np.zeros(ts.num_sites) + 0.01
            # H = ts.genotype_matrix().T
            # h = H[0].copy()
            # h[ts.num_sites // 2:] = H[-1, ts.num_sites // 2:]

            alleles = [var.alleles for var in ts.variants()]
            h = np.zeros(ts.num_sites, dtype=int)
            h[ts.num_sites // 2:] = 1
            np.random.seed(1)
            rho = np.random.random(ts.num_sites)
            mu = np.random.random(ts.num_sites) #* 0.1

            rho[0] = 0
            Ft, St = ls_forward_tree(h, alleles, ts, rho, mu)
            X = np.array([len(f) for f in Ft])
            Y = np.zeros_like(X)
            for j, f in enumerate(Ft):
                q = [round(v, 8) for v in f.values()]
                Y[j] = len(set(q))
            plt.plot(X)
            plt.plot(Y, label="distinct")
            plt.savefig("tmp/n={}_L={}.png".format(n, L))
            plt.clf()
            print(n, L, ts.num_sites, ts.num_trees, np.mean(X), np.mean(Y),
                    np.mean([len(a) for a in alleles]), sep="\t")


    # ts = msprime.simulate(250, recombination_rate=1, mutation_rate=2,
    #         random_seed=2, length=100)
    ts = msprime.simulate(
        8, recombination_rate=1, mutation_rate=2, random_seed=13, length=2)
    print("num_trees = ", ts.num_trees)
    # ts = jukes_cantor(ts, 200, 0.6, seed=1, multiple_per_node=False)

    # for h in ts.haplotypes():
    #     print(h)

    alleles = [var.alleles for var in ts.variants()]
    print("mean alleles = ", np.mean([len(a) for a in alleles]))
    H = ts.genotype_matrix().T
    n, m = H.shape
    # print("Shape = ", H.shape)
    h = np.zeros(ts.num_sites, dtype=int)
    h[ts.num_sites // 2:] = 1

    # h = H[0].copy()
    # h[ts.num_sites // 2:] = H[-1, ts.num_sites // 2:]
    h = H[-1]
    # H = H[:-1]
    rho = np.zeros(ts.num_sites) + 0.1
    mu = np.zeros(ts.num_sites) + 0.01

    # np.random.seed(1)
    # rho = np.random.random(ts.num_sites)
    # mu = np.random.random(ts.num_sites) #* 0.1
    # matrix_path, matrix_state = ls_matrix(h, H, rho, mu, return_internal=True)
    # print(H)
    # path, tree_state = ls_tree_naive(h, ts, rho, mu, return_internal=True)
#     hmm_path = np.array(model.decode(h))
#     print("matrix:", matrix_path)
#     print("hmm   :", hmm_path)
#     match = H[matrix_path, np.arange(ts.num_sites)]
#     print("match :", match)
#     print("h     :", h)

    rho[0] = 0
    model = ls_hmm(H, rho, mu)

    # F = ls_forward_matrix_unscaled(h, H, rho, mu)
    F, S = ls_forward_matrix(h, alleles, H, rho, mu)
    Ft, St = ls_forward_tree(h, alleles, ts, rho, mu)
    Ft = decode_ts_matrix(ts, Ft)

    assert np.allclose(S, St)
    assert np.allclose(F, Ft)

    log_prob = np.log(np.sum(F[:, -1])) - np.sum(np.log(S))
    print("log prob = ", log_prob)
    print("prob = ", np.exp(-log_prob))

    F *= np.cumprod(S)
    print("P = ", model.evaluate(h), np.sum(F[:, -1])) #/ np.prod(S))

#     # Only works for binary mutations.

#     alpha = model._forward(h)
#     Fp = np.zeros_like(F)
#     for j in range(len(h)):
#         Fp[:, j] = [alpha[j][k] for k in range(ts.num_samples)]
#         # print("site ", j)
#         # print(np.array([alpha[j][k] for k in range(ts.num_samples)]))
#         # print(F[:, j])
#         # print()

#     print("P = ", model.evaluate(h), np.sum(F[:, -1]))
#     assert np.allclose(F, Fp)

    # print(alpha[m - 1])

    # print(F)
    # print(Fp)

    # match = H[predict[1:], np.arange(ts.num_sites)]
    # print(h)
    # print(match)
    # print(logp, path)

    # logp, path = hmm.viterbi(h)
    # print(logp, path)

    # print(H.shape)
    # viz = CairoVisualiser(h, H, rho, mu, width=1800, height=1200)
    # viz.draw("output.png")

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
    # develop()
    # plot_encoding_efficiency()

    incremental_fitch_dev()


if __name__ == "__main__":
    main()
