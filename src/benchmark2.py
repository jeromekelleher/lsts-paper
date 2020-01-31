
import msprime
import tskit
import _tskit
import numpy as np
import time
import itertools
import collections
import os
import concurrent.futures
import random

import matplotlib as mp
# Force matplotlib to not use any Xwindows backend.
mp.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import humanize
import pandas as pd
import seaborn as sns
import tqdm
import sympy

import tests.test_li_stephens as tls

# ts = msprime.simulate(
#     10**5, Ne=10**4, length=100*10**6, mutation_rate=1e-8, recombination_rate=1e-8)
# ts.dump("benchmark.trees")

# ts = msprime.simulate(
#     10**6, Ne=10**4, length=100*10**6, mutation_rate=1e-8, recombination_rate=1e-8)
# ts.dump("benchmark-n1M-m100M.trees")

def subset_sites(ts, position):
    """
    Return a copy of the specified tree sequence with sites reduced to those
    with positions in the specified list.
    """
    tables = ts.dump_tables()
    lookup = frozenset(position)
    tables.sites.clear()
    tables.mutations.clear()
    for site in ts.sites():
        if site.position in lookup:
            site_id = tables.sites.add_row(
                site.position, ancestral_state=site.ancestral_state,
                metadata=site.metadata)
            for mutation in site.mutations:
                tables.mutations.add_row(
                    site_id, node=mutation.node, parent=mutation.parent,
                    derived_state=mutation.derived_state,
                    metadata=mutation.metadata)
    return tables.tree_sequence()



def main():
    # ts = tskit.load("benchmark.trees")
    # ts = tskit.load("benchmark-n100l-m1M.trees")
    # ts = tskit.load("benchmark-n1M-m100M.trees")
    # ts = tskit.load("tmp/2000000_panel.trees")
    # ts_queries = tskit.load("tmp/2000000_queries.trees")
    # ts = tskit.load("tmp/1390384_panel.trees")
    # ts_queries = tskit.load("tmp/1390384_queries.trees")

    # H = ts_queries.genotype_matrix().T

    # for j in range(1, 10):

    ts = msprime.simulate(
        25 * 10**1, Ne=10**4, length=10 * 10**6, mutation_rate=1e-8, recombination_rate=1e-8,
        random_seed=8)

    print("PID = ", os.getpid())
    print(ts.num_sites, ts.num_trees, ts.num_samples)
    # print("Matrix size = ", humanize.naturalsize(ts.num_sites * ts.num_samples * 4))
    G = ts.genotype_matrix()
    # print(H.shape)

    # simplified = ts.simplify([0, 1], filter_sites=False)
    # H = simplified.genotype_matrix().T

    rho = np.zeros(ts.num_sites) + 0.0125
    mu = np.zeros(ts.num_sites) + 0.125
    rho[0] = 0

    # N = np.zeros(m, dtype=int)
    # for j in range(m):
    #     N[j] = len(F[j])
    # print("mean = ", np.mean(N))
    # print("max = ", np.max(N))
    # print("N = ", N)

    precision = 7
    tolerance = 10**(-precision + 1)
    print("Precision = ", precision, "tolerance = ", tolerance)

    ls_hmm = _tskit.LsHmm(ts.ll_tree_sequence, rho, mu, precision=precision)
    fm = _tskit.ForwardMatrix(ts.ll_tree_sequence)

    print("running for ", ts.num_sites, "sites")
    for h in G.T:
        before = time.perf_counter()
        ls_hmm.forward_matrix(h, fm)
        S1 = fm.normalisation_factor
        duration = time.perf_counter() - before
        print("TREE: {:.2f}s".format(duration))
        before = time.perf_counter()
        F2, S2 = tls.ls_forward_matrix(h, [["0", "1"] for _ in range(ts.num_sites)], G, rho, mu)
        duration = time.perf_counter() - before
        print("Matrix: {:.2f}s".format(duration))
        print("S close", np.allclose(S1, S2, rtol=1, atol=tolerance))
        # F_decoded = tls.decode_ts_matrix(ts, F1)
        F_decoded = fm.decode()
        print("F close", np.allclose(F2, F_decoded, rtol=1, atol=tolerance))
        assert np.allclose(S1, S2, rtol=1, atol=tolerance)
        assert np.allclose(F2, F_decoded, rtol=1, atol=tolerance)


#         before = time.perf_counter()
#         N = ls_hmm.forward_matrix(h, True)
#         duration = time.perf_counter() - before

#         print("Done in {:.2f}s = {:.2f}ms/site N={:.2f} fraction N < 64 = {}".format(
#             duration, duration * 1e6 / ts.num_sites, np.mean(N),
#             np.sum(N < 64) / N.shape[0]))


def run(n, seed, rho, mu):
    ts = msprime.simulate(
        n, Ne=10**4, length=10 * 10**6, mutation_rate=1e-8, recombination_rate=1e-8,
        random_seed=seed)

    queries = [0, 1]
    panel = np.arange(2, n, dtype=np.int32)
    panel_ts = ts.simplify(panel)
    queries_ts = ts.simplify(queries, filter_sites=False)
    # Throw away any sites that are private to the query panel
    queries_ts = subset_sites(queries_ts, panel_ts.tables.sites.position)

    H = queries_ts.genotype_matrix().T
    m = panel_ts.num_sites

    rho = np.zeros(m) + rho
    mu = np.zeros(m) + mu
    rho[0] = 0
    precisions = [6, 8, 10, 12]
    ret = {"n": n, "seed": seed, "rho": rho[-1], "mu": mu[-1]}
    for precision in precisions:
        ls_hmm = _tskit.LsHmm(panel_ts.ll_tree_sequence, rho, mu, precision=precision)
        fm = _tskit.ForwardMatrix(panel_ts.ll_tree_sequence)
        ls_hmm.forward_matrix(H[0], fm)
        ret["p_{}".format(precision)] = np.mean(fm.num_transitions)
    return ret

def run_plot_distinct_values():

    fractions = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    random.seed(1)
    num_replicates = 5
    work = []
    for rho, mu in itertools.product(fractions, repeat=2):
        for n in np.logspace(1, 6, 20).astype(int):
            for _ in range(num_replicates):
                seed = random.randint(1, 2**31)
                work.append((n, seed, rho, mu))
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run, *w) for w in work]
        with tqdm.tqdm(total=len(futures)) as progress:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                df = pd.DataFrame(results)
                df.to_csv("distinct_values.csv")
                progress.update()


def plot_distinct_values():
    df = pd.read_csv("distinct_values.csv")

    for rho, mu in itertools.product(df.rho.unique(), df.mu.unique()):
        print("rho=", rho, "mu = ", mu)
        dfs = df[(df.rho == rho) & (df.mu == mu)]
        dfs = df.groupby(dfs.n).mean()
        for precision in [6, 8, 10, 12]:
            key = "p_{}".format(precision)
            # plt.semilogx(dfs.index, dfs[key], "-o", label=key)
            plt.semilogx(dfs.index, dfs[key] / (precision + 0), "-o",
                label="d = {}".format(precision))
            # plt.axvline(x=10**(precision / 2), ls="--", color="grey")
        plt.legend()
        plt.ylabel("K / d")
        plt.savefig("distinct_values_rho={}_mu={}.png".format(rho, mu))
        plt.clf()


def plot_distinct_values_random_matrix():
    m = 1000
    np.random.seed(1)
    rho = 1 / 4
    mu = 1 / 8
    ns = np.logspace(1, 4, 20).astype(int)
    # for n in
    N6 = np.zeros_like(ns, dtype=float)
    N8 = np.zeros_like(ns, dtype=float)
    for j, n in enumerate(ns):
        # G = np.random.randint(0, 2, size=(m, n))
        # h = np.random.randint(0, 2, size=m)

        G = np.zeros((m, n), dtype=int)
        for l in range(m):
            G[l, np.random.randint(n)] = 1
        h = np.zeros(m, dtype=int)

        rhov = np.zeros(m) + rho
        muv = np.zeros(m) + mu
        rhov[0] = 0
        F, S = tls.ls_forward_matrix(h, [["0", "1"] for _ in range(m)], G, rhov, muv)
        expr = (mu - (mu - 1) * (n - 1)) / n
        print(n, expr, S[0], S[-1])

        for N, precision in zip([N6, N8], [6, 8]):
            Fr = np.round(F, precision)
            v = np.zeros(m)
            for l in range(m):
                v[l] = np.unique(Fr[l]).shape[0]
            N[j] = np.mean(v)
    plt.semilogx(ns, N6 / 6, label="d = 6")
    plt.semilogx(ns, N8 / 8, label="d = 8")
    plt.xlabel("n")
    plt.ylabel("mean distinct values / precision")
    plt.legend()
    # plt.savefig("random_matrix.png")
    plt.savefig("singleton_matrix.png")



def examine_F_matrix():
    ts = msprime.simulate(
        10**3, Ne=10**4, length=10 * 10**6, mutation_rate=1e-8, recombination_rate=1e-8,
        random_seed=8)

    # # print("Matrix size = ", humanize.naturalsize(ts.num_sites * ts.num_samples * 4))


    for x in [4, 8]:
        m = ts.num_sites
        rho = np.zeros(m) + 1 / x
        mu = np.zeros(m) + 1 / x
        rho[0] = 0
        h = np.zeros(m, dtype=np.uint8) + 1
        G = ts.genotype_matrix()

        F, S = tls.ls_forward_matrix(h, [["0", "1"] for _ in range(m)], G, rho, mu)
        lp = tls.forward_matrix_log_proba(F, S)
        print(x, lp)
    # plt.savefig("hist.png")


    # print(F[:10])
    # print(F[50])
    # print(F[250])

    # column = np.atleast_2d(np.cumprod(S)).T
    # F = F * column

    # for row in F:
    #     print(row)

    # print(F_scaled)

    # Fr = np.round(F, 8)
    # for row in Fr:
    #     print(row)

    # for n in [4, 8, 12, 16]:
    #     Fr = np.round(F, n)
    #     N = np.array([np.unique(f).shape[0] for f in Fr])
    #     plt.plot(N, label=f"digits={n}")
    #     # N = np.array([np.sum(f == 0) / ts.num_samples for f in Fr])
    #     # print(N)
    # plt.legend()
    # plt.savefig("N.png")

    # fig, ax1 = plt.subplots()
    # Fr = np.round(F, 5)
    # N = np.array([np.unique(f).shape[0] for f in Fr])
    # ax1.plot(N)
    # ax2 = ax1.twinx()
    # f_var = np.var(F, axis=1)
    # ax2.plot(f_var, color="red")
    # plt.show()

    # print("S = ", np.log(S))
    # print("sum = ", np.cumsum(np.log(S)))

    # start = 22
    # stop = 50
    # start = 0
    # stop = 20
    # Fr = np.round(F, 12)
    # N = np.array([np.unique(f).shape[0] for f in Fr])
    # # plt.plot(range(start, stop), N[start:stop])
    # plt.plot(N)
    # plt.savefig("n.png")
    # plt.clf()

    # for j in range(start, stop):
    #     plt.hist(F[j], bins=100)
    #     # plt.hist(Fr[j], bins=100)
    #     plt.savefig(f"site={j}.png")
    #     plt.clf()

    # # F = Fr
    # df = pd.DataFrame({
    #     "f": np.hstack([F[j] for j in range(start, stop)]),
    #     "site": np.hstack([np.zeros(ts.num_samples) + j for j in range(start, stop)])})
    # # print(df)

    # ax = sns.boxplot(x="site", y="f", data=df)
    # plt.savefig("valuedist.png")
    # # plt.show()

def sites_for_mismatch_decay(rho, mu, n, eps):
    """
    Returns expected number of sites downstream from a mismatch with
    the specified recombination and mutation probabilities until the
    forward matrix value is within eps of 1 / n. This is for a
    singleton reference panel (i.e., each site is a singleton).
    """
    return np.log((mu - 1) * n * eps / (2 * mu - 1)) / np.log(1 - rho)


def value_after_mismatch(rho, mu, n, k):
    """
    Return the value of the F matrix k sites after a mismatch on a particular
    haplotype. (Assuming singleton matrix).
    """
    z = (1 - rho)**k
    return (mu * z / (1 - mu) - z + 1) / n


def test_mismatch_decay():
    n = 1000
    m = 10000
    rho = 1 / 32
    mu = 1 / 8

    G = np.zeros((m, n), dtype=int)
    G[0, 0] = 1
    # for l in range(m):
        # G[l, l % n] = 1
    h = np.zeros(m, dtype=int)

    rhov = np.zeros(m) + rho
    muv = np.zeros(m) + mu
    rhov[0] = 0
    F, S = tls.ls_forward_matrix(h, [["0", "1"] for _ in range(m)], G, rhov, muv)
    print(S, 1 - mu)
    eps = 1e-6
    expected_k = sites_for_mismatch_decay(rho, mu, n, eps)
    print("expected k = ", expected_k)
    k_values = np.zeros(m - 20)
    for l in range(m):
    # for l in [0]:
        if l >= k_values.shape[0]:
            break
        mismatch = l % n
        # print(mismatch)
        for k in range(m - l):
            # print(k, F[l + k, mismatch], value_after_mismatch(rho, mu, n, k))
            if 1 / n - F[l + k, mismatch] <= eps:
                k_values[l] = k
                break
        # print(F[l])
        # print("l = ", l, k_values[l])
        # print("\t", F[l + k, mismatch], F[l + k, mismatch] - F[l + k, mismatch - 1])
    print(k_values[0])
    print(np.mean(k_values))




def sympy_F_matrix():

    n = 10000
    mu = 1 / 8
    rho = 1 / 4

    def g(k):
        return (1 - (1 - rho)**k * (1 - mu)) / n

    def f(x):
        return x * (1 - rho) + rho / n

    ksol = np.log(1e-6 / (1 - mu)) / np.log(1 - rho)
    print("iterations needed for eps = ", ksol)
    for k in range(10):
        eps = (1 - rho)**k * (1 - mu)
        print(k, sites_for_mismatch_decay(rho, mu, eps), eps)

#     x = mu / n
#     for k in range(30):
#         print(k, x, g(k))
#         x = f(x)
#         # print(k, ((1 - rho)**k) * (1 - mu))


#     fs = [f(k) for k in range(20)]
#     plt.plot(fs)
#     plt.axhline(y=1 / n, color="grey")
#     plt.savefig("converging.png")


#     rho = sympy.symbols("rho", positive=True, real=True)
#     mu = sympy.symbols("mu", positive=True, real=True)
#     n = sympy.symbols("n")
#     c = sympy.symbols("c")
#     z = sympy.symbols("z")
#     k = sympy.symbols("k")
#     eps = sympy.symbols("eps", positive=True, real=True)

#     # expr = (1 - (1 - rho)**k * (1 - mu)) / n
#     expr = (1 - rho)**k * (1 - mu)
#     eq = sympy.Eq(expr, eps)
#     print("eq = ", eq)
#     sol = sympy.solveset(eq, k, domain=sympy.S.Reals)
#     print("sol = ", sol)
#     # def f(x):
#     #     return x * (1 - rho) + rho * c

    # def fn(x, k):
    #     return x * (1 - rho)**k + (((1 - rho)**k - 1) / (1 - rho - 1)) * (rho * c)

#     expr = fn(mu / n, k).subs(c, 1 - mu).simplify()
#     print(expr)

    # # x = z
    # # for j in range(10):
    # #     print(x.expand())
    # #     x = f(x)

#     n = 1000
#     # mu = 1 / 8
#     rho = 1 / 8

#     def f(x):
#         return x * (1 - rho) + rho / n

#     c = 1 / n
#     def fn(x, k):
#         # return x * (1 - rho)**k + (((1 - rho)**k - 1) / (1 - rho - 1)) * (rho * c)
#         return c - c * (1 - rho)**k + x * (1 - rho)**k

#     x = 2 / n
#     z = x
#     v = []
#     u = []
#     for j in range(30):
#         v.append(z)
#         u.append(fn(x, j))
#         print(j, z, u[-1])
#         z = f(z)

#     plt.plot(v)
#     plt.plot(u)
#     plt.axhline(y=1 / n, color="grey")
#     plt.savefig("converging.png")


    # S = (mu - (mu - 1) * (n - 1)) / n
    # print("S = ", S, 1 - mu)


#     z = 1 / n
#     mismatch = (z * (1 - rho) + rho / n) * mu / S
#     match = (z * (1 - rho) + rho / n) * (1 - mu) / S

#     def next_site(F):
#         return (F * (1 - rho) + rho / n) * (1 - mu) / S

#     print(S)
#     u = []
#     v = []
#     for j in range(25):
#         # print(match, mismatch, mismatch / match, sep="\t")
#         # u.append(match)
#         v.append(match - mismatch)
#         x  = (2 * j * mu * rho - j * rho - mu) / (mu * n - n)
#         print(mismatch, x)
#         match = next_site(match)
#         mismatch = next_site(mismatch)

#     # plt.plot(u)
#     plt.plot(v)
#     plt.savefig("converging.png")

#     sympy.init_printing()

#     mu = sympy.symbols("mu")
#     rho = sympy.symbols("rho")
#     n = sympy.symbols("n")
#     z = 1 / n
#     S = (mu - (mu - 1) * (n - 1)) / n
#     # Approximate this to 1 - mu for large n
#     S = 1 - mu

#     # # This is the expression for S[0] when we have a reference panel
#     # # of singletons
#     # expr = (
#     #     ((1 - rho) / n + rho / n) * (1 - mu) * (n - 1) +
#     #     ((1 - rho) / n + rho / n) * mu)
#     # print(expr.simplify())

#     # mismatch = ((z * (1 - rho) + rho / n) * mu / S)
#     # match = ((z * (1 - rho) + rho / n) * (1 - mu) / S)
#     mismatch = mu / n
#     match = 1 / n

#     def next_site(F):
#         return (F * (1 - rho) + rho / n) * (1 - mu) / S
#     values = {mu: 1 / 8, rho : 1 / 2,  n: 1000}
#     # values[S] = expr.subs(values)
#     print(values)

#     for j in range(6):
#         print("j = ", j)
#         print("mismatch = ", mismatch.expand())
#         print("match    = ", match)
#         print(mismatch.subs(values))
#         print(match.subs(values))
#         match = next_site(match).simplify()
#         mismatch = next_site(mismatch).simplify()


    # print(next_site(mismatch))
    # print(match)

    # print(mu)
    # G = np.array([
    #     [1, 0, 0],
    #     [0, 0, 1],
    #     [1, 0, 1],
    #     [0, 1, 1],
    #     [0, 1, 1],

    #     [1, 0, 1],
    #     ])
    # m, n = G.shape
    # h = [0, 0, 0, 0, 0, 0]


    # F = [None for _ in range(m)]
    # S = [None for _ in range(m)]
    # f = [1 for _ in range(n)]
    # # Use this to compare with real calculations.
    # # f = [1 / n for _ in range(n)]
    # for l in range(m):
    #     f = [f[j] * (mu if h[l] != G[l, j] else 1 - mu) for j in range(n)]
    #     S[l] = sum(f)
    #     for j in range(n):
    #         f[j] /= S[l]
    #     F[l] = list(f)
    # for l in range(m):
    #     print()
    #     print("l = ", l)
    #     print("S = ", sympy.simplify(S[l] + sympy.O(mu**3)))
    #     for j in range(n):
    #         print("\t", sympy.simplify(F[l][j] + sympy.O(mu**3)))


    # rho = np.zeros(m)
    # mu = np.zeros(m) + 0.0125
    # F, S = tls.ls_forward_matrix(h, [["0", "1"] for _ in range(m)], G, rho, mu)
    # # print(F)
    # print(S)

def expected_allele_frequency():
    ns = np.arange(1, 20) * 1000
    num_reps = 1000
    F = np.zeros((ns.shape[0], num_reps), dtype=float)
    for j, n in enumerate(ns):
        print(n)
        for k, ts in enumerate(msprime.simulate(n, mutation_rate=1, num_replicates=num_reps)):
            f = np.zeros(ts.num_sites)
            for tree in ts.trees():
                for site in tree.sites():
                    u = site.mutations[0].node
                    f[site.id] = tree.num_samples(u)
            f /= n
            F[j, k] = np.mean(f)
    Fbar = np.mean(F, axis=1)
    plt.plot(ns, Fbar)
    plt.show()


if __name__ == "__main__":
    # main()
    # examine_F_matrix()
    # run_plot_distinct_values()
    # plot_distinct_values()
    # plot_distinct_values_random_matrix()
    # sympy_F_matrix()
    test_mismatch_decay()
    # expected_allele_frequency()

