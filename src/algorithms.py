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
import attr

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


def fitch_sets_from_mutations(tree, mutations):
    """
    Given a tree and a set of mutations on the tree, return the corresponding
    Fitch sets for all nodes that are ancestral to mutations.
    """
    def compute(u, parent_state):
        child_sets = []
        for v in tree.children(u):
            # If the set for a given child is empty, then we know it inherits
            # directly from the parent state and must be a singleton set.
            if len(A[v]) == 0:
                child_sets.append({parent_state})
            else:
                child_sets.append(A[v])
        A[u] = set.intersection(*child_sets)
        if len(A[u]) == 0:
            A[u] = set.union(*child_sets)

    A = [set() for _ in range(tree.num_nodes)]
    for u in sorted(mutations.keys(), key=lambda u: tree.time(u)):
        # Compute the value at this node
        if tree.is_internal(u):
            compute(u, mutations[u])
        else:
            A[u] = {mutations[u]}
        # Find parent state
        v = tree.parent(u)
        if v != -1:
            while v not in mutations:
                v = tree.parent(v)
            parent_state = mutations[v]
            v = tree.parent(u)
            while v not in mutations:
                compute(v, parent_state)
                v = tree.parent(v)
    return A

def incremental_fitch_sets_from_mutations(tree, mutations):
    """
    Given a tree and a set of mutations on the tree, return the corresponding
    Fitch sets for all nodes that are ancestral to mutations. This is done
    in incremental manner using just the parent pointer in the tree.
    """
    # This isn't working either.
    # Very tricky to work out how to propagate the values around the tree
    ts = tree.tree_sequence
    values = np.unique(list(mutations.values()))
    K = values.shape[0]
    A = np.zeros((ts.num_nodes, K), dtype=int)
    U = np.zeros((ts.num_nodes, K), dtype=int)
    N = np.zeros(ts.num_nodes, dtype=int)
    paths = np.zeros(ts.num_nodes, dtype=int)
    mu = np.zeros(ts.num_nodes, dtype=int) - 1
    for u, value in mutations.items():
        mu[u] = np.searchsorted(values, value)
    parent = [tree.parent(u) for u in range(ts.num_nodes)]
    for u in tree.nodes():
        N[u] = len(tree.children(u))

    node_labels = {u: f"{u} " for u in tree.nodes()}
    node_labels.update({u: f"{u}->{mu[u]}" for u in mutations.keys()})
    print(tree.draw(format="unicode", node_labels=node_labels))

    # First count the paths upwards from each mutation node
    for u in mutations.keys():
        # paths[u] = 1
        v = tree.parent(u)
        if v != -1:
            while mu[v] == -1 and paths[v] == 0:
                paths[v] = 1
                v = parent[v]
            paths[v] += 1

    stack = [(tree.root, mu[tree.root])]
    while len(stack) > 0:
        u, state = stack.pop()
        if mu[u] == -1:
            U[u, state] = paths[u]
        print("VISIT", u, paths[u], N[u])
        for v in tree.children(u):
            if mu[v] != -1:
                stack.append((v, mu[v]))
            elif paths[v] > 0:
                stack.append((v, state))

    print("INITIAL")
    node_labels = {u: f"{u}:{paths[u]}:{U[u]}" for u in tree.nodes()}
    print(tree.draw(format="unicode", node_labels=node_labels))
#     # Now set the initial U values
#     for u in mutations.keys():
#         v = parent[u]
#         while mu[v] == -1:


    def a(u):
        a = U[u] == N[u]
        if np.sum(a) == 0 or N[u] == 0:
            a = U[u] > 0
        return a.astype(int)

    def propagate_loss(node):
        v = node
        u = parent[v]
        while u != -1:
            U[u] -= A[v]
            v = u
            u = parent[u]

    def propagate_gain(node):
        v = node
        u = parent[v]
        while u != -1:
            U[u] += A[v]
            A[u] = a(u)
            v = u
            u = parent[u]

    # Now propagate set values upwards for each
    for u in mutations.keys():
        v = parent[u]
        if v != -1:
            # while mu[v] == -1:
            #     v = parent[v]
            # parent_state = mu[v]
            # print("mutation = ", u, parent_state)
            # propagate_loss(u)
            U[u, mu[u]] += 1
            propagate_gain(u)

    node_labels = {u: f"{u}:{paths[u]}:{U[u]}" for u in tree.nodes()}
    print(tree.draw(format="unicode", node_labels=node_labels))



    A2 = [set(values[np.where(a == 1)[0]]) for a in A]
    return A2



@attr.s
class MutationTreeNode(object):
    tree_node = attr.ib(default=-1)
    parent = attr.ib(default=-1)
    child = attr.ib(default=-1)
    sib = attr.ib(default=-1)
    num_children = attr.ib(default=0)
    index = attr.ib(default=-1)


def fitch_sets_from_mutations_by_embedding(tree, mutations):
    """
    Given a tree and a set of mutations on the tree, return the corresponding
    Fitch sets for all nodes that are ancestral to mutations. This is done
    by first constructing an embedded mutation tree.


    This would be a neat idea if we could make it work, but it ends up being
    quite complex. The embedded tree in the worst case is as complicated as the
    real tree, even though most of the time it's a small number of nodes on top
    of the mutations. The main problem is that we can get to a situation where
    we must have mutations over the anonymous dangling nodes which are
    supposed to represent the intermediate tree nodes.

    Might be worth revisiting, as it seems like it should work, so not deleting
    for now. The attractive thing about this is that we should be able to
    fit the entire mutation tree in cache, and it would be far more efficient to
    run Fitch on this embedded tree rather than the full tree.
    """

    node_labels = {u: f"{u} " for u in tree.nodes()}
    node_labels.update({u: f"{u}:{mutations[u]}" for u in mutations})
    print(tree.draw(format="unicode", node_labels=node_labels))

    mutation_nodes = sorted(mutations.keys(), key=lambda u: tree.time(u))
    x = np.zeros(tree.tree_sequence.num_nodes, dtype=int) - 1
    mutation_tree = [
        MutationTreeNode(tree_node=u, index=j) for j, u in enumerate(mutation_nodes)]
    for j, u in enumerate(mutation_nodes):
        x[u] = j
    for j, u in enumerate(mutation_nodes[:-1]):
        # print("mutation:", j, "tree_node = ", u)
        mtn = mutation_tree[j]
        u = tree.parent(u)
        assert u != -1
        while x[u] == -1:
            # print("\tsetting ", u, "->", mtn.index)
            x[u] = mtn.index
            u = tree.parent(u)
        # print("Finished traversal, u = ", u, x[u], ":", mtn)
        if mutation_tree[x[u]].tree_node == u:
            # We've hit an existing node in the tree.
            mtn.parent = x[u]
            # print("\tHit existing internal", u, mtn)
        elif u != -1:
            # print("\tCreating new join node at ", u)
            # Need to create a new node.
            new_mtn = MutationTreeNode(tree_node=u, index=len(mutation_tree))
            mutation_tree.append(new_mtn)
            new_mtn.parent = mutation_tree[x[u]].parent
            mutation_tree[x[u]].parent = new_mtn.index
            mtn.parent = new_mtn.index
            # print("\tmtn = ", mtn)
            # print("\tnew_mtn = ", new_mtn)
            # Propagate the new node upwards.
            x_old = x[u]
            while x[u] == x_old:
                x[u] = new_mtn.index
                u = tree.parent(u)

    # If there are edges in the mutation tree which span more than a single edge
    # in the overall tree, insert a join node and dangling leaf to represent all
    # of these subtrees in the Fitch sets.
    for j in range(len(mutation_tree)):
        mtn = mutation_tree[j]
        if mtn.parent != -1:
            parent_mtn = mutation_tree[mtn.parent]
            u = tree.parent(mtn.tree_node)
            nodes = []
            if u != parent_mtn.tree_node:
                nodes.append(u)
                u = tree.parent(u)
                if u != parent_mtn.tree_node:
                    nodes.append(u)
            # print("nodes = ", nodes)
            for u in nodes:
                # print("INSERT JOIN and DANGLE", u)
                join_mtn = MutationTreeNode(
                    tree_node=u, index=len(mutation_tree), parent=parent_mtn.index)
                mutation_tree.append(join_mtn)
                mtn.parent = join_mtn.index
                leaf_mtn = MutationTreeNode(
                    parent=join_mtn.index, index=len(mutation_tree))
                mutation_tree.append(leaf_mtn)
                # print(mtn)
                # print(parent_mtn)
                x_old = x[u]
                while u != -1 and x[u] == x_old:
                    x[u] = join_mtn.index
                    u = tree.parent(u)
                mtn = join_mtn

    # Update the triply-linked tree.
    for mtn in mutation_tree:
        if mtn.parent != -1:
            parent_mtn = mutation_tree[mtn.parent]
            mtn.sib = parent_mtn.child
            parent_mtn.child = mtn.index
            parent_mtn.num_children += 1

    for mtn in mutation_tree:
        if mtn.num_children == 1:
            print("UNARY NODE", mtn, mtn.tree_node)
            new_mtn = MutationTreeNode(
                index=len(mutation_tree), parent=mtn.index, sib=mtn.child)
            mutation_tree.append(new_mtn)
            mtn.child = new_mtn.index
            mtn.num_children += 1
    print("size of mutation tree = ", len(mutation_tree))

    # node_labels = {u: f"{u}:{x[u]}" for u in tree.nodes()}
    # print(tree.draw(format="unicode", node_labels=node_labels))
    tables = tskit.TableCollection(1)
    for j, mtn in enumerate(mutation_tree):
        time = 0 if mtn.tree_node == -1 else tree.time(mtn.tree_node)
        tables.nodes.add_row(time=time, flags=1)
        if mtn.parent != -1:
            tables.edges.add_row(0, 1, parent=mtn.parent, child=j)
        # print(mtn)
    tables.sort()
    ts = tables.tree_sequence()
    embedded_t = ts.first()
    node_labels = {j: f"{j}:{mtn.tree_node}" for j, mtn in enumerate(mutation_tree)}
    print(embedded_t.draw(format="unicode", node_labels=node_labels))

    stack = [len(mutations) - 1]
    while len(stack) > 0:
        j = stack.pop()
        children = []
        k = mutation_tree[j].child
        while k != -1:
            children.append(k)
            k = mutation_tree[k].sib
        assert sorted(children) == sorted(embedded_t.children(j))
        # print("NODE ", j, children, embedded_t.children(j))
        stack.extend(children)


    A = [set() for _ in mutation_tree]
    for j in embedded_t.nodes(order="postorder"):
        # print("VISIT", j, mutation_tree[j])
        u = mutation_tree[j].tree_node
        if embedded_t.is_internal(j):
            A[j] = set.intersection(*[A[k] for k in embedded_t.children(j)])
            if len(A[j]) == 0:
                A[j] = set.union(*[A[k] for k in embedded_t.children(j)])
            for k in embedded_t.children(j):
                mtn = mutation_tree[k]
                # if mtn.tree_node == -1:
                #     print("FIXED CHILD", k, A[k], " A = ", A[j])
                #     A[j] = A[k]
        else:
            # print("LEAF", mutation_tree[j])
            k = j
            while u not in mutations:
                k = embedded_t.parent(k)
                u = mutation_tree[k].tree_node
            A[j] = {mutations[u]}
            # print("resolved to ", A[j])


    node_labels = {j: f"{j}:{A[j]}" for j, mtn in enumerate(mutation_tree)}
    print(embedded_t.draw(format="unicode", node_labels=node_labels))

    # Now traverse down to compute the parsimonious assigments on the
    # mutation tree.
    root = len(mutations) - 1
    ancestral_state = list(A[root])[0]
    f = {mutation_tree[root].tree_node: ancestral_state}
    stack = [(root, ancestral_state)]
    while len(stack) > 0:
        j, state = stack.pop()
        k = mutation_tree[j].child
        while k != -1:
            child_state = state
            if state not in A[k]:
                child_state = list(A[k])[0]
                u = mutation_tree[k].tree_node
                print("u = ", u, "k = ", k)
                assert u != -1
                f[u] = child_state
            stack.append((k, child_state))
            k = mutation_tree[k].sib

    A2 = [None for _ in range(tree.tree_sequence.num_nodes)]
    for j, mtb in enumerate(mutation_tree):
        if mtb.tree_node != -1:
            A2[mtb.tree_node] = A[j]
    return A2, f


def get_parsimonious_mutations(tree, mutations):
    """
    Returns a parsimonious set of mutations by using mutation-oriented Fitch
    parsimony.
    """
    A = fitch_sets_from_mutations(tree, mutations)
    old_state = mutations[tree.root]
    new_state = list(A[tree.root])[0]
    f = {tree.root: new_state}
    stack = [(tree.root, old_state, new_state)]
    while len(stack) > 0:
        u, old_state, new_state = stack.pop()
        # print("VISIT", u, old_state, new_state)
        for v in tree.children(u):
            old_child_state = old_state
            if v in mutations:
                old_child_state = mutations[v]
            if len(A[v]) > 0:
                new_child_state = new_state
                if new_state not in A[v]:
                    new_child_state = list(A[v])[0]
                    f[v] = new_child_state
                stack.append((v, old_child_state, new_child_state))
            else:
                if old_child_state != new_state:
                    f[v] = old_child_state

                # print("SKIP", v, old_child_state, new_state)
    return f


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
            for v in tree.children(u):
                A[u] += A[v] > 0
            k = len(tree.children(u))
            if np.any(A[u] == k):
                A[u, A[u] < k] = 0
    return A


def incremental_fitch_counts(ts, labels):
    parent = np.zeros(ts.num_nodes, dtype=int) - 1
    K = np.max(labels) + 1
    A = np.zeros((ts.num_nodes, K), dtype=int)
    U = np.zeros((ts.num_nodes, K), dtype=int)
    N = np.zeros(ts.num_nodes, dtype=int)

    for label, sample in zip(labels, ts.samples()):
        A[sample, label] = 1

    # We'd like to remove the A array entirely, but we need the old values of A
    # when we're propagating upwards and the dependencies are tricky. Possibly
    # this can be done if we propagate losses up first for all edges, but it's
    # not trivial.
    def a(u):
        a = U[u] == N[u]
        if np.sum(a) == 0 or N[u] == 0:
            a = U[u] > 0
        return a.astype(int)

    def propagate_loss(node):
        v = node
        u = parent[v]
        while u != -1:
            U[u] -= A[v]
            v = u
            u = parent[u]

    def propagate_gain(node):
        v = node
        u = parent[v]
        while u != -1:
            U[u] += A[v]
            A[u] = a(u)
            v = u
            u = parent[u]

    for (left, right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            propagate_loss(edge.child)
            parent[edge.child] = -1
            N[edge.parent] -= 1
            A[edge.parent] = a(edge.parent)
            propagate_gain(edge.parent)

        for edge in edges_in:
            parent[edge.child] = edge.parent
            N[edge.parent] += 1
            propagate_loss(edge.parent)
            propagate_gain(edge.child)

        yield A


def random_mutations(tree, num_mutations, num_labels, seed):
    """
    Returns a random set of mutations on the tree. These may include non-mutations
    in which we transition to the same state. The root will always have a mutation,
    and so the returned dictionary will have num_mutations + 1 items.
    """
    rng = random.Random(seed)
    f = {tree.root: rng.randint(0, num_labels)}
    nodes = list(tree.nodes())
    for _ in range(min(num_mutations, len(nodes) - 1)):
        node = tree.root
        while node in f:
            node = rng.choice(nodes)
        f[node] = rng.randint(0, num_labels)
    return f

def project_genotypes(tree, mutations, dtype=np.uint8):
    genotypes = np.zeros(tree.tree_sequence.num_samples, dtype=dtype)
    for j, u in enumerate(tree.tree_sequence.samples()):
        while u not in mutations:
            u = tree.parent(u)
        genotypes[j] = mutations[u]
    return genotypes


def get_parsimonious_mutations_by_projection(tree, mutations):
    genotypes = project_genotypes(tree, mutations)
    ancestral_state, (node, _, state) = tree.reconstruct(genotypes)
    f = {tree.root: ancestral_state}
    f.update(dict(zip(node, state)))
    return f

def dynamic_fitch_single_tree(tree, num_mutations=3, num_labels=3, seed=1):
    f1 = random_mutations(tree, num_mutations, num_labels=num_labels, seed=seed)
    f2 = get_parsimonious_mutations_by_projection(tree, f1)
    g2 = project_genotypes(tree, f2)
    f3 = get_parsimonious_mutations(tree, f1)
    g3 = project_genotypes(tree, f3)
    A1 = fitch_sets(tree, g2)
    A2 = fitch_sets_from_mutations(tree, f1)

    # A3 = incremental_fitch_sets_from_mutations(tree, f1)

    # assert A2 == A3

    # A3, f4 = fitch_sets_from_mutations_by_embedding(tree, f1)
    # g4 = project_genotypes(tree, f4)

    # print("====================")
    # print("A = ", A1)
    # print("A = ", A3)
    # node_labels = {u: str(A1[u]) for u in tree.nodes()}
    # print(tree.draw(format="unicode", node_labels=node_labels))
    # node_labels = {u: str(A2[u]) for u in tree.nodes()}
    # print(tree.draw(format="unicode", node_labels=node_labels))
    # node_labels = {u: str(A3[u]) for u in tree.nodes()}
    # print(tree.draw(format="unicode", node_labels=node_labels))
    # for f in [f1]: #[f1, f2, f3]:
    #     node_labels = {u: f"{u}  " for u in tree.nodes()}
    #     node_labels.update({u: f"{u}:{state}" for u, state in f.items()})
    #     print(f)
    #     print(tree.draw(format="unicode", node_labels=node_labels))
    # print(g2)
    # print(g3)
    # print(len(f1), len(f2), len(f3))

    for u in f1:
        assert A1[u] == A2[u]
    assert len(f2) == len(f3)
    assert np.array_equal(g2, g3)


def incremental_fitch_dev():
    ts = msprime.simulate(8, recombination_rate=0, random_seed=2)
    # dynamic_fitch_single_tree(ts.first(), num_mutations=5, num_labels=5, seed=3)
    for n in range(3, 100):
        ts = msprime.simulate(n, recombination_rate=0, random_seed=2)
        for num_mutations in range(min(50, n)):
        # for num_mutations in [10]:
            for num_labels in range(2, 10):
                # print("===============", "muts = ", num_mutations, "labels = ", num_labels)
                print(n, num_mutations, num_labels)
                for seed in range(1, 4):
                    dynamic_fitch_single_tree(
                        ts.first(), num_mutations=num_mutations, num_labels=num_labels, seed=seed)


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

def compress(tree, mutations):

    A = [set() for _ in range(tree.num_nodes)]
    for u in sorted(mutations.keys(), key=lambda u: tree.time(u)):
        # State of mutation is always in node set.
        mutation_state = mutations[u]
        A[u].add(mutation_state)
        # Find parent state
        v = tree.parent(u)
        while v != -1 and v not in mutations:
            v = tree.parent(v)
        if v != -1:
            parent_state = mutations[v]
            u = tree.parent(u)
            while u != -1:
                child_sets = []
                for v in tree.children(u):
                    # If the set for a given child is empty, then we know it inherits
                    # directly from the parent state and must be a singleton set.
                    if len(A[v]) == 0:
                        child_sets.append({parent_state})
                    else:
                        child_sets.append(A[v])
                A[u] = set.intersection(*child_sets)
                if len(A[u]) == 0:
                    A[u] = set.union(*child_sets)
                u = tree.parent(u)

    old_state = mutations[tree.root]
    new_state = list(A[tree.root])[0]
    f = {tree.root: new_state}
    stack = [(tree.root, old_state, new_state)]
    while len(stack) > 0:
        u, old_state, new_state = stack.pop()
        # print("VISIT", u, old_state, new_state)
        for v in tree.children(u):
            old_child_state = old_state
            if v in mutations:
                old_child_state = mutations[v]
            if len(A[v]) > 0:
                new_child_state = new_state
                if new_state not in A[v]:
                    new_child_state = list(A[v])[0]
                    f[v] = new_child_state
                stack.append((v, old_child_state, new_child_state))
            else:
                if old_child_state != new_state:
                    f[v] = old_child_state

                # print("SKIP", v, old_child_state, new_state)

    N = {u: tree.num_samples(u) for u in f}
    for u in sorted(f.keys(), key=lambda u: -tree.time(u)):
        v = tree.parent(u)
        while v != tskit.NULL and v not in f:
            v = tree.parent(v)
        if v != tskit.NULL:
            N[v] -= N[u]

    return f, N


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
        # Choose a value arbitrarily and promote its value to root.
        u, x = f.popitem()
        while u != tree.root:
            u = tree.parent(u)
            assert u not in f
        f[u] = x
        f, N = compress(tree, f)
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
                f[u] = round(p_t * p_e, 8)

            f, N = compress(tree, f)

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



def ls_forward_tree(h, alleles, ts, mu, rho, precision=10):
    """
    Forward matrix computation based on a tree sequence.
    """
    fa = ForwardAlgorithm(ts, rho, mu, precision=precision)
    # fa = OldForwardAlgorithm(ts, rho, mu, precision=precision)
    # fa = ForwardAlgorithmMutationTree(ts, rho, mu, precision=precision)
    return fa.run(h, alleles)


@attr.s
class ProbabilityMutation(object):
    node = attr.ib(default=-1)
    probability = attr.ib(default=-1)
    num_samples = attr.ib(default=0)
    # Triply linked tree. Each of these refers to the index in a list of nodes.
    parent = attr.ib(default=-1)
    child = attr.ib(default=-1)
    sib = attr.ib(default=-1)

# This seemed like a nice idea at, so keeping it lying around for now. It's a
# partial implementatation of the forward algorithm which stores the mutations
# in an embedded tree. Where this does reduce the need to do upwards traversals
# to determine the state of a probability, it's a lot of extra complexity. In
# particular, maintaining the embedded tree requires a lot of traversing around
# in the first place during tree transitions, so it's not at all clear it would
# be faster.
class ForwardAlgorithmMutationTree(object):
    """
    Runs the Li and Stephens forward algorithm.
    """
    def __init__(self, ts, mu, rho, precision=10):
        self.ts = ts
        self.mu = mu
        self.rho = rho
        self.precision = precision
        n, m = ts.num_samples, ts.num_sites
        # The output F matrix. Each site is a dictionary containing a compressed
        # probability array.
        self.F = [None for _ in range(m)]
        # The output normalisation array.
        self.S = np.zeros(m)
        # The list of ProbabilityMutations
        self.P = []
        # The index of the ProbabilityMutation associated with a given node, or -1.
        self.Q = np.zeros(ts.num_nodes, dtype=int) - 1

    def check_integrity(self, tree):
        print(self.draw_tree(tree))
        Q_copy = self.Q.copy()
        num_nodes = 0
        for j, pm in enumerate(self.P):
            print(pm)
            if pm.node != -1:
                assert self.Q[pm.node] == j
                assert pm.probability >= 0
                Q_copy[pm.node] = -1
                num_nodes += 1
        assert np.all(Q_copy == -1)

        visited_nodes = 0
        for tree_root in tree.roots:
            root = self.Q[tree_root]
            assert root != -1
            # Rebuild the mutation tree.
            mutation_tree = collections.defaultdict(list)
            stack = [(tree_root, root)]
            while len(stack) > 0:
                tree_node, parent_mutation = stack.pop()
                for child in tree.children(tree_node):
                    child_mutation = parent_mutation
                    if self.Q[child] != -1:
                        child_mutation = self.Q[child]
                        mutation_tree[parent_mutation].append(child_mutation)
                    stack.append((child, child_mutation))

            # Traverse the mutation tree.
            stack = [root]
            while len(stack) > 0:
                j = stack.pop()
                pm = self.P[j]
                visited_nodes += 1
                local_children = []
                child = pm.child
                while child != -1:
                    local_children.append(child)
                    pmc = self.P[child]
                    assert pmc.parent == j
                    stack.append(child)
                    child = pmc.sib
                print("children", local_children, mutation_tree[j])
                assert sorted(local_children) == sorted(mutation_tree[j])
        assert visited_nodes == num_nodes

    def draw_tree(self, tree):
        node_labels = {u: f"{u}  " for u in tree.nodes()}
        for pm in self.P:
            if pm.node != -1:
                node_labels[pm.node] = "{} :{:.3f}=({}, {}, {})".format(
                    pm.node, pm.probability, pm.parent, pm.child, pm.sib)
        return tree.draw(format="unicode", node_labels=node_labels)


    def duplicate_probability_mutation(self, tree, tree_node):
        """
        Inserts a new node into the probability mutation tree at the specified
        node which carries the same value as it currently inherits
        """
        print("Insert duplicate", tree_node)
        print(self.draw_tree(tree))
        P = self.P
        Q = self.Q
        # Find the next probability mutation above us in the tree.
        u = tree_node
        while Q[u] == tskit.NULL:
            u = tree.parent(u)
        parent_id = Q[u]
        parent_pm = P[parent_id]
        # For each of the children, check if the current node exists on the path
        # from the child mutation to the parent.
        child_id = parent_pm.child
        last_child_pm = None
        while child_id != tskit.NULL:
            child_pm = P[child_id]
            u = P[child_id].node
            while u != parent_pm.node and u != tree_node:
                u = tree.parent(u)
            if u == tree_node:
                break
            last_child_pm = child_pm
            child_id = P[child_id].sib
        # Create the new mutation
        q = len(P)
        Q[tree_node] = q
        print("PARENT = ", parent_pm)
        print("child_id = ", child_id)
        pm = ProbabilityMutation(
            node=tree_node, probability=parent_pm.probability, parent=parent_id)
        if child_id != tskit.NULL:
            pm.sib = child_pm.sib
            child_pm.sib = -1
            child_pm.parent = q
            pm.child = child_id
        if last_child_pm is not None:
            last_child_pm.sib = q
        else:
            parent_pm.child = q
        P.append(pm)

        print("DONE")
        print(self.draw_tree(tree))

    def remove_edge(self, tree, edge):
        print("REMOVE EDGE")
        if self.Q[edge.child] == -1:
            self.duplicate_probability_mutation(tree, edge.child)
        root = self.Q[edge.child]
        # Now we need to break the tree rooted at this node out of the
        # overall mutation tree so that it can be grafted back in somewhere
        # else
        root_pm = self.P[root]
        assert root_pm.parent != -1
        parent_pm = self.P[root_pm.parent]
        print("parent = ", parent_pm)
        sib_id = parent_pm.child
        last_sib_id = -1
        while sib_id != root:
            print("Root = ", root_pm, "sib = ", sib_id)
            last_sib_id = sib_id
            sib_id = self.P[sib_id].sib
        if last_sib_id == -1:
            parent_pm.child = root_pm.sib
        else:
            last_sib_pm = self.P[last_sib_id]
            last_sib_pm.sib = root_pm.sib
        root_pm.sib = -1
        root_pm.parent = -1

    def insert_edge(self, tree, edge):
        print("INSERT", edge)
        Q = self.Q
        P = self.P
        # TODO need to repair parent/sib/etc connections here.
        if Q[edge.parent] == -1:
            # If the parent has no mutation, this is the first child to be added, so
            # we move the mutation up one node.
            q = Q[edge.child]
            Q[edge.child] = -1
            Q[edge.parent] = q
            P[q].node = edge.parent
        elif P[Q[edge.parent]].probability == P[Q[edge.child]].probability:
            # If the parent and the child have mutations with the same value
            # then delete the child mutation.
            P[Q[edge.child]].node = -1
            Q[edge.child] = -1


    def compress(self, tree):
        print(self.draw_tree(tree))

        self.check_integrity(tree)
        Q = self.Q
        P = self.P
        A = [set() for _ in range(self.ts.num_nodes)]
        # Post-order traversal of the mutation tree.
        stack = [Q[tree.root]]
        k = tskit.NULL
        while len(stack) > 0:
            j = stack[-1]
            child = P[j].child
            if child != tskit.NULL and j != k:
                while child != tskit.NULL:
                    stack.append(child)
                    child = P[child].sib
            else:
                k = P[j].parent
                stack.pop()
                if k != tskit.NULL:
                    # Visit the probability mutation at j
                    pm = P[j]
                    # State of mutation is always in node set.
                    A[pm.node].add(pm.probability)
                    parent_state = P[k].probability
                    u = tree.parent(pm.node)
                    while u != -1:
                        child_sets = []
                        for v in tree.children(u):
                            # If the set for a given child is empty, then we know it inherits
                            # directly from the parent state and must be a singleton set.
                            if len(A[v]) == 0:
                                child_sets.append({parent_state})
                            else:
                                child_sets.append(A[v])
                        A[u] = set.intersection(*child_sets)
                        if len(A[u]) == 0:
                            A[u] = set.union(*child_sets)
                        u = tree.parent(u)


        print(tree.draw(format="unicode", node_labels={u: str(A[u]) for u in tree.nodes()}))

        f = {pm.node: pm.probability for pm in P if pm.node != -1}
        A2 = fitch_sets_from_mutations(tree, f)
        assert A == A2

        for pm in P:
            if pm.node != -1:
                Q[pm.node] = -1
        assert np.all(Q == -1)
        P.clear()

        old_state = f[tree.root]
        new_state = list(A[tree.root])[0]
        P.append(ProbabilityMutation(node=tree.root, probability=new_state))
        Q[tree.root] = 0
        stack = [(tree.root, old_state, new_state, 0)]
        while len(stack) > 0:
            u, old_state, new_state, parent_mutation = stack.pop()
            print("VISIT", u, old_state, new_state, parent_mutation)
            for v in tree.children(u):
                print("\t", v, new_state, A[v], new_state in A[v])
                old_child_state = old_state
                mutation_required = False
                if v in f:
                    old_child_state = f[v]
                if len(A[v]) > 0:
                    new_child_state = new_state
                    child_parent_mutation = parent_mutation
                    if new_state not in A[v]:
                        mutation_required = True
                        new_child_state = list(A[v])[0]
                        # Actual mutation is added below
                        child_parent_mutation = len(P)
                    stack.append((v, old_child_state, new_child_state, child_parent_mutation))
                else:
                    if old_child_state != new_state:
                        mutation_required = True
                        new_child_state = old_child_state

                print("\t", v, "new_child_state =  ", new_child_state)
                if mutation_required:
                    print("\tADDING MUTATION", v)
                    parent_pm = P[parent_mutation]
                    pm = ProbabilityMutation(
                        node=v, probability=new_child_state, parent=parent_mutation,
                        sib=parent_pm.child)
                    parent_pm.child = len(P)
                    self.Q[v] = len(P)
                    P.append(pm)

        self.check_integrity(tree)

        f_dict = get_parsimonious_mutations(tree, f)
        print(f_dict)
        print({pm.node: pm.probability for pm in P})
        assert f_dict == {pm.node: pm.probability for pm in P}

        for pm in P:
            pm.num_samples = tree.num_samples(pm.node)

        for pm in P:
            # Subtract the number of samples subtended by each child
            child = pm.child
            while child != -1:
                child_pm = P[child]
                pm.num_samples -= child_pm.num_samples
                child = child_pm.sib

        self.check_integrity(tree)


    def run(self, h, alleles):
        n = self.ts.num_samples
        Q = self.Q
        P = self.P

        for j, u in enumerate(self.ts.samples()):
            P.append(ProbabilityMutation(node=u, probability=1 / n))
            Q[u] = j

        tree = tskit.Tree(self.ts)
        for (left, right), edges_out, edges_in in self.ts.edge_diffs():
            # print("start", left, right, M)
            self.check_integrity(tree)
            print("BEFORE")
            print(self.draw_tree(tree))
            for edge in edges_out:
                self.remove_edge(tree, edge)
            for edge in edges_in:
                self.insert_edge(tree, edge)

            tree.next()
            print("AFTER")
            print(self.draw_tree(tree))
            self.check_integrity(tree)

            # print("NEW TREE")
            # after = draw_tree(tree, f)
            # for l1, l2 in zip(before.splitlines(), after.splitlines()):
            #     print(l1, "|", l2)

            for site in tree.sites():
                l = site.id
                print("l = ", l, h[l], site.mutations)
                # print("P = ", self.P)
                self.check_integrity(tree)
                for mutation in site.mutations:
                    if Q[mutation.node] == -1:
                        self.duplicate_probability_mutation(tree, mutation.node)
                self.check_integrity(tree)

                mutations = {mut.node: mut.derived_state for mut in site.mutations}
                for pm in P:
                    u = pm.node
                    if u != tskit.NULL:
                        assert pm.probability >= 0
                        # Get the state at u. TODO we can add a state_cache here.
                        v = u
                        while v != tskit.NULL and v not in mutations:
                            v = tree.parent(v)
                        allele = mutations.get(v, site.ancestral_state)
                        state = alleles[site.id].index(allele)

                        # Compute the F value for u.
                        p_t = pm.probability * (1 - self.rho[l]) + self.rho[l] / n
                        p_e = self.mu[l]
                        if h[l] == state:
                            p_e = 1 - (len(alleles[l]) - 1) * self.mu[l]
                        pm.probability = round(p_t * p_e, self.precision)
                        assert pm.probability >= 0

                self.compress(tree)
                # Normalise and store
                self.S[l] = sum(pm.probability * pm.num_samples for pm in P)
                for pm in P:
                    pm.probability /= self.S[l]
                self.F[l] = {pm.node: pm.probability for pm in P}

                # self.S[l] = sum(N[u] * f[u] for u in M)
                # for u in M:
                #     f[u] /= self.S[l]
                # # print("f = ", f)
                # self.F[l] = {u: f[u] for u in M}

        return self.F, self.S


class ForwardAlgorithm(object):
    """
    Runs the Li and Stephens forward algorithm.
    """
    def __init__(self, ts, mu, rho, precision=10):
        self.ts = ts
        self.mu = mu
        self.rho = rho
        self.precision = precision
        n, m = ts.num_samples, ts.num_sites
        # The output F matrix. Each site is a dictionary containing a compressed
        # probability array.
        self.F = [None for _ in range(m)]
        # The output normalisation array.
        self.S = np.zeros(m)
        # The probablilites associated with each mutation.
        self.f = np.zeros(ts.num_nodes) - 1
        # List of nodes containing mutations.
        self.M = []
        # Number of samples directly inheriting from each mutation
        self.N = np.zeros(ts.num_nodes, dtype=int)
        self.parent = np.zeros(ts.num_nodes, dtype=int) - 1

    def check_integrity(self):
        assert np.all(self.f[self.M] >= 0)
        index = np.ones_like(self.f, dtype=bool)
        index[self.M] = 0
        assert np.all(self.f[index] == -1)

    def draw_tree(self, tree):
        node_labels = {u: f"{u}  " for u in tree.nodes()}
        for u in self.M:
            node_labels[u] = "{} :{:.3f}".format(u, self.f[u])
        return tree.draw(format="unicode", node_labels=node_labels)

    def compress(self, tree):
        self.check_integrity()
        M = self.M
        f = self.f
        values = np.unique(list(f[u] for u in M))

        def compute(u, parent_state):
            union = np.zeros(len(values), dtype=int)
            inter = np.ones(len(values), dtype=int)
            child = np.zeros(len(values), dtype=int)
            for v in tree.children(u):
                child[:] = A[v]
                # If the set for a given child is empty, then we know it inherits
                # directly from the parent state and must be a singleton set.
                if np.sum(child) == 0:
                    child[parent_state] = 1
                union = np.logical_or(union, child)
                inter = np.logical_and(inter, child)
            if np.sum(inter) > 0:
                A[u] = inter
            else:
                A[u] = union

        A = np.zeros((tree.tree_sequence.num_nodes, len(values)), dtype=int)

        M.sort(key=lambda u: tree.time(u))
        for u in M:
            # Compute the value at this node
            state = np.searchsorted(values, f[u])
            if tree.is_internal(u):
                compute(u, state)
            else:
                A[u, state] = 1
            # Find parent state
            v = tree.parent(u)
            if v != -1:
                while f[v] == -1:
                    v = tree.parent(v)
                parent_state = np.searchsorted(values, f[v])
                v = tree.parent(u)
                while f[v] == -1:
                    compute(v, parent_state)
                    v = tree.parent(v)

        A2 = [{values[j] for j in np.where(row == 1)[0]} for row in A]
        A3 = fitch_sets_from_mutations(tree, {u: f[u] for u in M})
        # print(tree.draw(format="unicode"))
        # for u, (r0, r1, r2) in enumerate(zip(A, A2, A3)):
        #     print(u, r0, r1, r2, sep="\t")
        assert A2 == A3

        f_copy = f.copy()
        f[M] = -1
        M.clear()
        old_state = np.searchsorted(values, f_copy[tree.root])
        new_state = np.where(A[tree.root] == 1)[0][0]
        f[tree.root] = values[new_state]
        M.append(tree.root)
        stack = [(tree.root, old_state, new_state)]
        while len(stack) > 0:
            u, old_state, new_state = stack.pop()
            for v in tree.children(u):
                old_child_state = old_state
                if f_copy[v] != -1:
                    old_child_state = np.searchsorted(values, f_copy[v])
                if np.sum(A[v]) > 0:
                    new_child_state = new_state
                    if A[v, new_state] == 0:
                        new_child_state = np.where(A[v] == 1)[0][0]
                        # new_child_state = list(A[v])[0]
                        f[v] = values[new_child_state]
                        M.append(v)
                    stack.append((v, old_child_state, new_child_state))
                else:
                    if old_child_state != new_state:
                        f[v] = values[old_child_state]
                        M.append(v)

        self.N[:] = 0
        for u in self.M:
            self.N[u] = tree.num_samples(u)
        for u in self.M:
            v = tree.parent(u)
            while v != tskit.NULL and self.f[v] == -1:
                v = tree.parent(v)
            if v != tskit.NULL:
                self.N[v] -= self.N[u]

        self.check_integrity()

    def run(self, h, alleles):
        n = self.ts.num_samples
        f = self.f
        S = self.S
        M = self.M
        N = self.N
        parent = self.parent

        for u in self.ts.samples():
            f[u] = 1 / n
            M.append(u)

        tree = tskit.Tree(self.ts)
        for (left, right), edges_out, edges_in in self.ts.edge_diffs():
            # print("start", left, right, M)
            self.check_integrity()
            g1 = project_genotypes(tree, {u: f[u] for u in M}, dtype=np.float64)

            before = self.draw_tree(tree)
            for edge in edges_out:
                # print("REMOVE", edge)
                u = edge.child
                if f[u] == -1:
                    # Make sure the subtree we're detaching has an f-value at the root.
                    while f[u] == -1:
                        u = parent[u]
                    f[edge.child] = f[u]
                    M.append(edge.child)
                parent[edge.child] = -1
            self.check_integrity()

            for edge in edges_in:
                # print("INSERT", edge)
                parent[edge.child] = edge.parent

                if parent[edge.parent] == -1:
                    # Grafting onto a new root.
                    if f[edge.parent] == -1:
                        f[edge.parent] = f[edge.child]
                        M.append(edge.parent)
                    if f[edge.parent] == f[edge.child]:
                        f[edge.child] = -1
                        M.remove(edge.child)
                else:
                    # Grafting into an existing subtree.
                    u = edge.parent
                    while f[u] == -1:
                        u = parent[u]
                    assert u != -1
                    if f[u] == f[edge.child]:
                        f[edge.child] = -1
                        M.remove(edge.child)

            self.check_integrity()

            tree.next()
            g2 = project_genotypes(tree, {u: f[u] for u in M}, dtype=np.float64)
            # print("NEW TREE")
            # after = self.draw_tree(tree)
            # for l1, l2 in zip(before.splitlines(), after.splitlines()):
            #     print(l1, "|", l2)
            # print(g1)
            # print(g2)
            assert np.all(g1 == g2)

            for site in tree.sites():
                l = site.id
                # print("l = ", l, h[l], site.mutations)
                # print("M = ", M)
                # print("f = ", {u: f[u] for u in M})
                assert np.all(f[M] >= 0)
                for mutation in site.mutations:
                    u = mutation.node
                    while u != tskit.NULL and f[u] == tskit.NULL:
                        u = tree.parent(u)
                    if f[mutation.node] == -1:
                        M.append(mutation.node)
                    f[mutation.node] = 0 if u == tskit.NULL else f[u]

                mutations = {mut.node: mut.derived_state for mut in site.mutations}
                for u in M:
                    assert f[u] >= 0
                    # Get the state at u. TODO we can add a state_cache here.
                    v = u
                    while v != tskit.NULL and v not in mutations:
                        v = tree.parent(v)
                    allele = mutations.get(v, site.ancestral_state)
                    state = alleles[site.id].index(allele)

                    # Compute the F value for u.
                    p_t = f[u] * (1 - self.rho[l]) + self.rho[l] / n
                    p_e = self.mu[l]
                    if h[l] == state:
                        p_e = 1 - (len(alleles[l]) - 1) * self.mu[l]
                    f[u] = round(p_t * p_e, self.precision)
                    assert f[u] >= 0

                self.compress(tree)
                # Normalise and store
                self.S[l] = sum(N[u] * f[u] for u in M)
                for u in M:
                    f[u] /= self.S[l]
                # print("f = ", f)
                self.F[l] = {u: f[u] for u in M}

        return self.F, self.S




class OldForwardAlgorithm(object):
    """
    Runs the Li and Stephens forward algorithm.
    """
    def __init__(self, ts, mu, rho, precision=10):
        self.ts = ts
        self.mu = mu
        self.rho = rho
        self.precision = precision
        n, m = ts.num_samples, ts.num_sites
        # The output F matrix. Each site is a dictionary containing a compressed
        # probability array.
        self.F = [None for _ in range(m)]
        # The output normalisation array.
        self.S = np.zeros(m)
        # The probablilites associated with each mutation.
        self.f = np.zeros(ts.num_nodes) - 1
        # List of nodes containing mutations.
        self.M = []
        # Number of samples directly inheriting from each mutation
        self.N = np.zeros(ts.num_nodes, dtype=int)
        self.parent = np.zeros(ts.num_nodes, dtype=int) - 1

    def check_integrity(self):
        assert np.all(self.f[self.M] >= 0)
        index = np.ones_like(self.f, dtype=bool)
        index[self.M] = 0
        assert np.all(self.f[index] == -1)

    def draw_tree(self, tree):
        node_labels = {u: f"{u}  " for u in tree.nodes()}
        for u in self.M:
            node_labels[u] = "{} :{:.3f}".format(u, self.f[u])
        return tree.draw(format="unicode", node_labels=node_labels)

    def simple_compress(self, tree):
        self.check_integrity()
        f_dict = get_parsimonious_mutations(tree, {u: self.f[u] for u in self.M})
        self.f[self.M] = -1
        self.M.clear()
        for u, value in f_dict.items():
            self.f[u] = value
            self.M.append(u)

        self.N[:] = 0
        for u in self.M:
            self.N[u] = tree.num_samples(u)
        for u in self.M:
            v = tree.parent(u)
            while v != tskit.NULL and self.f[v] == -1:
                v = tree.parent(v)
            if v != tskit.NULL:
                self.N[v] -= self.N[u]

    def compress(self, tree):
        self.check_integrity()
        M = self.M
        f = self.f

        def compute(u, parent_state):
            child_sets = []
            for v in tree.children(u):
                # If the set for a given child is empty, then we know it inherits
                # directly from the parent state and must be a singleton set.
                if len(A[v]) == 0:
                    child_sets.append({parent_state})
                else:
                    child_sets.append(A[v])
            A[u] = set.intersection(*child_sets)
            if len(A[u]) == 0:
                A[u] = set.union(*child_sets)

        A = [set() for _ in range(tree.tree_sequence.num_nodes)]
        M.sort(key=lambda u: tree.time(u))
        for u in M:
            # Compute the value at this node
            if tree.is_internal(u):
                compute(u, f[u])
            else:
                A[u] = {f[u]}
            # Find parent state
            v = tree.parent(u)
            if v != -1:
                while f[v] == -1:
                    v = tree.parent(v)
                parent_state = f[v]
                v = tree.parent(u)
                while f[v] == -1:
                    compute(v, parent_state)
                    v = tree.parent(v)

        assert A == fitch_sets_from_mutations(tree, {u: f[u] for u in M})

        f_copy = f.copy()
        f[M] = -1
        M.clear()
        old_state = f_copy[tree.root]
        new_state = list(A[tree.root])[0]
        f[tree.root] = new_state
        M.append(tree.root)
        stack = [(tree.root, old_state, new_state)]
        while len(stack) > 0:
            u, old_state, new_state = stack.pop()
            # print("VISIT", u, old_state, new_state)
            for v in tree.children(u):
                old_child_state = old_state
                if f_copy[v] != -1:
                    old_child_state = f_copy[v]
                if len(A[v]) > 0:
                    new_child_state = new_state
                    if new_state not in A[v]:
                        new_child_state = list(A[v])[0]
                        f[v] = new_child_state
                        M.append(v)
                    stack.append((v, old_child_state, new_child_state))
                else:
                    if old_child_state != new_state:
                        f[v] = old_child_state
                        M.append(v)

        self.N[:] = 0
        for u in self.M:
            self.N[u] = tree.num_samples(u)
        for u in self.M:
            v = tree.parent(u)
            while v != tskit.NULL and self.f[v] == -1:
                v = tree.parent(v)
            if v != tskit.NULL:
                self.N[v] -= self.N[u]

        self.check_integrity()

    def run(self, h, alleles):
        n = self.ts.num_samples
        f = self.f
        S = self.S
        M = self.M
        N = self.N
        parent = self.parent

        for u in self.ts.samples():
            f[u] = 1 / n
            M.append(u)

        tree = tskit.Tree(self.ts)
        for (left, right), edges_out, edges_in in self.ts.edge_diffs():
            # print("start", left, right, M)
            self.check_integrity()
            g1 = project_genotypes(tree, {u: f[u] for u in M}, dtype=np.float64)

            before = self.draw_tree(tree)
            for edge in edges_out:
                # print("REMOVE", edge)
                u = edge.child
                if f[u] == -1:
                    # Make sure the subtree we're detaching has an f-value at the root.
                    while f[u] == -1:
                        u = parent[u]
                    f[edge.child] = f[u]
                    M.append(edge.child)
                parent[edge.child] = -1
            self.check_integrity()

            for edge in edges_in:
                # print("INSERT", edge)
                parent[edge.child] = edge.parent

                if parent[edge.parent] == -1:
                    # Grafting onto a new root.
                    if f[edge.parent] == -1:
                        f[edge.parent] = f[edge.child]
                        M.append(edge.parent)
                    if f[edge.parent] == f[edge.child]:
                        f[edge.child] = -1
                        M.remove(edge.child)
                else:
                    # Grafting into an existing subtree.
                    u = edge.parent
                    while f[u] == -1:
                        u = parent[u]
                    assert u != -1
                    if f[u] == f[edge.child]:
                        f[edge.child] = -1
                        M.remove(edge.child)

            self.check_integrity()

            tree.next()
            g2 = project_genotypes(tree, {u: f[u] for u in M}, dtype=np.float64)
            # print("NEW TREE")
            # after = self.draw_tree(tree)
            # for l1, l2 in zip(before.splitlines(), after.splitlines()):
            #     print(l1, "|", l2)
            # print(g1)
            # print(g2)
            assert np.all(g1 == g2)

            for site in tree.sites():
                l = site.id
                # print("l = ", l, h[l], site.mutations)
                # print("M = ", M)
                # print("f = ", {u: f[u] for u in M})
                assert np.all(f[M] >= 0)
                for mutation in site.mutations:
                    u = mutation.node
                    while u != tskit.NULL and f[u] == tskit.NULL:
                        u = tree.parent(u)
                    if f[mutation.node] == -1:
                        M.append(mutation.node)
                    f[mutation.node] = 0 if u == tskit.NULL else f[u]

                mutations = {mut.node: mut.derived_state for mut in site.mutations}
                for u in M:
                    assert f[u] >= 0
                    # Get the state at u. TODO we can add a state_cache here.
                    v = u
                    while v != tskit.NULL and v not in mutations:
                        v = tree.parent(v)
                    allele = mutations.get(v, site.ancestral_state)
                    state = alleles[site.id].index(allele)

                    # Compute the F value for u.
                    p_t = f[u] * (1 - self.rho[l]) + self.rho[l] / n
                    p_e = self.mu[l]
                    if h[l] == state:
                        p_e = 1 - (len(alleles[l]) - 1) * self.mu[l]
                    f[u] = round(p_t * p_e, self.precision)
                    assert f[u] >= 0

                self.compress(tree)
                # Normalise and store
                self.S[l] = sum(N[u] * f[u] for u in M)
                for u in M:
                    f[u] /= self.S[l]
                # print("f = ", f)
                self.F[l] = {u: f[u] for u in M}

        return self.F, self.S


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

    alleles = [var.alleles for var in ts.variants()]

    for h, mu, rho in itertools.product(haplotypes, mus, rhos):
        rho[0] = 0
        F, S = ls_forward_matrix(h, alleles, H, rho, mu)
        # FIXME we need some way of computing what the expected precision loss
        # is here and accounting for it.
        Ft, St = ls_forward_tree(h, alleles, ts, rho, mu, precision=100)
        Ft = decode_ts_matrix(ts, Ft)

        assert np.allclose(S, St)
        assert np.allclose(F, Ft)



def verify_worker(work):
    n, length = work
    ts = msprime.simulate(
        n, recombination_rate=0.1, mutation_rate=2,
            random_seed=12, length=length)
    verify_tree_algorithm(ts)
    return ts.num_samples, ts.num_sites, ts.num_trees


def verify():
    work = itertools.product([3, 5, 20, 50], [1, 10, 100])
    # for w in work:
    #     print("Verify ", w)
    #     verify_worker(w)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_work = {executor.submit(verify_worker, w): w for w in work}
        for future in concurrent.futures.as_completed(future_to_work):
            n, m, t = future.result()
            print("Verify n =", n, "num_sites =", m, "num_trees", t, flush=True)

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
            Ft, St = ls_forward_tree(h, alleles, ts, rho, mu, precision=10)
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

def develop():

    # ts = msprime.simulate(250, recombination_rate=1, mutation_rate=2,
    #         random_seed=2, length=100)
    ts = msprime.simulate(
        8, recombination_rate=1, mutation_rate=3, random_seed=13, length=2)
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

    rho[0] = 0
    model = ls_hmm(H, rho, mu)

    # F = ls_forward_matrix_unscaled(h, H, rho, mu)
    F, S = ls_forward_matrix(h, alleles, H, rho, mu)
    # Ft, St = ls_forward_tree_naive(h, alleles, ts, rho, mu)
    Ft, St = ls_forward_tree(h, alleles, ts, rho, mu)
    Ft = decode_ts_matrix(ts, Ft)

    # print(S)
    # print(St)
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
    develop()
    # plot_encoding_efficiency()

    # incremental_fitch_dev()


if __name__ == "__main__":
    main()
