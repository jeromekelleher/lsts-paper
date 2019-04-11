"""
Run benchmarks for Li and Stephens algorithms.
"""

import os
import subprocess
import pathlib
import concurrent.futures
import functools
import gzip
import time
import sys

sys.path.insert(0, "../tskit/python") # TMP
import _tskit

import msprime
import tskit
import numpy as np
import pandas as pd
import matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datadir = pathlib.Path("data")
bulk_datadir = datadir / "bulk__NOBACKUP__"

panel_sizes = np.logspace(3, 6, 20).astype(int) * 2

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


def write_bcf(ts, filename):
    read_fd, write_fd = os.pipe()
    write_pipe = os.fdopen(write_fd, "w")
    with open(filename, "w") as bcf_file:
        proc = subprocess.Popen(
            ["bcftools", "view", "-O", "b"], stdin=read_fd, stdout=bcf_file)
    ts.write_vcf(write_pipe, ploidy=2)
    write_pipe.close()
    os.close(read_fd)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("bcftools failed with status:", proc.returncode)

def split_ts(ts, num_queries):
    """
    Splits the specified tree sequence into pair of panel and query 
    tree sequences.
    """
    # first we must edit the tree sequence to change the alleles to 0/1 to 
    # workaround a current limitation.
    m = ts.num_sites
    assert ts.num_mutations == m
    tables = ts.dump_tables()
    tables.mutations.set_columns(
        site=tables.mutations.site,
        node=tables.mutations.node,
        derived_state=np.zeros_like(tables.mutations.derived_state) + ord("1"),
        derived_state_offset=np.arange(m + 1, dtype=np.uint32))

    tables.sites.set_columns(
        position=tables.sites.position,
        ancestral_state=np.zeros_like(tables.sites.ancestral_state) + ord("0"),
        ancestral_state_offset=np.arange(m + 1, dtype=np.uint32))
    ts = tables.tree_sequence()

    panel_size = ts.num_samples - num_queries
    panel = np.arange(panel_size, dtype=np.int32) + num_queries
    queries = np.arange(num_queries, dtype=np.int32)
    panel_ts = ts.simplify(panel)
    queries_ts = ts.simplify(queries, filter_sites=False)
    # Throw away any sites that are private to the query panel
    queries_ts = subset_sites(queries_ts, panel_ts.tables.sites.position)
    return panel_ts, queries_ts


def run_simulation(panel_size, num_queries=1000, length=100):
    print("Running n = ", panel_size)
    ts = msprime.simulate(
        panel_size + num_queries, Ne=10**4, recombination_rate=1e-8,
        mutation_rate=1e-8, random_seed=42, length=length * 10**6)
    ts.dump(str(bulk_datadir / "{}_full.trees".format(panel_size)))
    panel_ts, queries_ts = split_ts(ts, num_queries)
    panel_ts.dump(str(bulk_datadir / "{}_panel.trees".format(panel_size)))
    queries_ts.dump(str(bulk_datadir / "{}_queries.trees".format(panel_size)))
    if panel_size < 50000:
        # vcf_file = str(bulk_datadir / "{}_panel.vcf.gz".format(panel_size))
        # with gzip.open(vcf_file, "wt") as f:
        #     ts.write_vcf(f, ploidy=2)
        write_bcf(ts, str(bulk_datadir / "{}_panel.bcf".format(panel_size)))
    return panel_size


def run_simulations():
    num_queries = 1000

    # run_simulation(panel_size[0], num_queries, length=1)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, n, num_queries, length=100) for n in panel_sizes]
        for future in concurrent.futures.as_completed(futures):
            print("Completed", future.result())

def check_simulations():
    for n in panel_sizes:
        panel_ts = tskit.load(str(bulk_datadir / "{}_panel.trees".format(n)))
        queries_ts = tskit.load(str(bulk_datadir / "{}_queries.trees".format(n)))
        assert np.array_equal(panel_ts.tables.sites.position, queries_ts.tables.sites.position)
        print(n, panel_ts.num_sites, queries_ts.num_sites)

def benchmark_similar_haplotypes(panel_ts, queries_ts, num_queries=2, precision=8):
    H = queries_ts.genotype_matrix().T
    m = panel_ts.num_sites
    recombination_rate = np.zeros(m) + 0.125
    mutation_rate = np.zeros(m) + 0.0125
    ls_hmm = _tskit.LsHmm(
        panel_ts.ll_tree_sequence,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate, precision=precision)
    # Need a big block size here or we trigger assertions on UKB.
    fm = _tskit.ForwardMatrix(panel_ts.ll_tree_sequence, block_size=2**20)
    num_values = np.zeros(num_queries)
    cpu_time = np.zeros(num_queries)
    for j in range(num_queries):
        before = time.perf_counter()
        ls_hmm.forward_matrix(H[j], fm)
        cpu_time[j] = time.perf_counter() - before
        num_values[j] = np.mean(fm.num_transitions)
        print("Query ", j, "in ", cpu_time[j], "= ", cpu_time[j] / panel_ts.num_sites)
    return {
        "num_values": np.mean(num_values), "cpu_time": np.mean(cpu_time),
        "num_sites": panel_ts.num_sites, "num_samples": panel_ts.num_samples}


def benchmark_random_haplotypes(panel_ts, num_queries=2, seed=1, precision=8):
    np.random.seed(seed)
    m = panel_ts.num_sites
    recombination_rate = np.zeros(m) + 0.125
    mutation_rate = np.zeros(m) + 0.0125
    ls_hmm = _tskit.LsHmm(
        panel_ts.ll_tree_sequence,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate, precision=precision)
    num_values = np.zeros(num_queries)
    cpu_time = np.zeros(num_queries)
    for j in range(num_queries):
        h = np.random.randint(0, 2, size=m, dtype=np.uint8)
        before = time.perf_counter()
        result = ls_hmm.forward_matrix(h, True)
        cpu_time[j] = time.perf_counter() - before
        num_values[j] = np.mean(result)
    return {
        "num_values": np.mean(num_values), "cpu_time": np.mean(cpu_time),
        "num_sites": panel_ts.num_sites, "num_samples": panel_ts.num_samples}


def run_benchmarks():
    rows = []
    num_replicates = 5
    for n in panel_sizes:
        panel_ts = tskit.load(str(bulk_datadir / "{}_panel.trees".format(n)))
        queries_ts = tskit.load(str(bulk_datadir / "{}_queries.trees".format(n)))
        result = benchmark_random_haplotypes(panel_ts, num_replicates)
        result["type"] = "random"
        print(result)
        rows.append(result)
        result = benchmark_similar_haplotypes(panel_ts, queries_ts, num_replicates)
        result["type"] = "similar"
        print(result)
        rows.append(result)

        df = pd.DataFrame(rows)
        df.to_csv("data/benchmark.csv")

def run_data_benchmarks():
    # The nosimplify trees are smaller, but the algorithm doesn't seem to work
    # on them. We get weird results, where it thinks there's only a few thousand sites.
    # filename = "../treeseq-inference/human-data/ukbb_chr20.augmented_131072.trees"
    filename = "../treeseq-inference/human-data/1kg_chr20.trees"
    #ts = tskit.load(
    ts = tskit.load(filename)
    print("loaded", filename)
    panel_ts, queries_ts = split_ts(ts, 2)
    result = benchmark_similar_haplotypes(panel_ts, queries_ts, precision=6)
    print(result)
    print(result["cpu_time"] / result["num_sites"] * 10**6, "mu s")
    # We're fair bit off the pace here. On cycloid we get 

    # loaded ../treeseq-inference/human-data/1kg_chr20.trees
    # Query  0 in  125.48219530284405 =  0.00014582474759191637
    # Query  1 in  127.97423614561558 =  0.00014872078575899544
    # {'num_values': 206.35909529343405, 'cpu_time': 126.72821572422981, 'num_sites': 860500, 'num_samples': 5006}
    # 147.27276667545593 mu s

    # loaded ../treeseq-inference/human-data/ukbb_chr20.augmented_131072.trees
    # Query  0 in  57.604527808725834 =  0.003647241218736598
    # Query  1 in  69.05904776230454 =  0.004372486245555562
    # {'num_values': 36.64951880460934, 'cpu_time': 63.33178778551519, 'num_sites': 15794, 'num_samples': 974652}
    # 4009.8637321460806 mu s


def plot_benchmarks():
    df = pd.read_csv("data/benchmark.csv")
    df.cpu_time /= df.num_sites
    # Convert to microseconds
    df.cpu_time *= 1e6

    df_random = df.query("type == 'random'")
    df_similar= df.query("type == 'similar'")
    plt.semilogx(df_random.num_samples, df_random.cpu_time, "-o", label="Random haplotypes")
    plt.semilogx(df_similar.num_samples, df_similar.cpu_time, "-o", label="Similar haplotypes")
    plt.ylabel("CPU Time (microseconds)")
    plt.xlabel("Reference panel size")
    plt.legend()
    plt.savefig("cpu_time.png")
    plt.clf()

    plt.semilogx(df_random.num_samples, df_random.num_values, "-o", label="Random haplotypes")
    plt.semilogx(df_similar.num_samples, df_similar.num_values, "-o", label="Similar haplotypes")
    plt.ylabel("Mean probabilities on tree")
    plt.xlabel("Reference panel size")
    plt.legend()
    plt.savefig("num_values.png")
    plt.clf()


def main():
    # run_simulations()
    # check_simulations()
    # run_benchmarks()
    # plot_benchmarks()
    run_data_benchmarks()

if __name__ == "__main__":
    main()
