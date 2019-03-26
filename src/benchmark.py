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

def run_simulation(panel_size, num_queries=1000, length=100):
    print("Running n = ", panel_size)
    ts = msprime.simulate(
        panel_size + num_queries, Ne=10**4, recombination_rate=1e-8,
        mutation_rate=1e-8, random_seed=42, length=length * 10**6)
    ts.dump(str(bulk_datadir / "{}_full.trees".format(panel_size)))
    panel = np.arange(panel_size, dtype=np.int32) + num_queries
    queries = np.arange(num_queries, dtype=np.int32)
    panel_ts = ts.simplify(panel)
    panel_ts.dump(str(bulk_datadir / "{}_panel.trees".format(panel_size)))
    queries_ts = ts.simplify(queries, filter_sites=False)
    # Throw away any sites that are private to the query panel
    queries_ts = subset_sites(queries_ts, panel_ts.tables.sites.position)
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

def benchmark_tskit(panel_ts, queries_ts, num_queries=2):
    H = queries_ts.genotype_matrix().T
    m = panel_ts.num_sites
    recombination_rate = np.zeros(m) + 0.125
    mutation_rate = np.zeros(m) + 0.0125
    ls_hmm = _tskit.LsHmm(
        panel_ts.ll_tree_sequence,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate, precision=10)
    num_values = np.zeros(num_queries)
    cpu_time = np.zeros(num_queries)
    for j in range(num_queries):
        before = time.perf_counter()
        result = ls_hmm.forward_matrix(H[j], True)
        cpu_time[j] = time.perf_counter() - before
        num_values[j] = np.mean(result)
    return {
        "num_values": np.mean(num_values), "cpu_time": np.mean(cpu_time),
        "num_sites": panel_ts.num_sites, "num_samples": panel_ts.num_samples}

def run_benchmarks():
    rows = []
    for n in panel_sizes:
        panel_ts = tskit.load(str(bulk_datadir / "{}_panel.trees".format(n)))
        queries_ts = tskit.load(str(bulk_datadir / "{}_queries.trees".format(n)))
        result = benchmark_tskit(panel_ts, queries_ts, 5)
        rows.append(result)
        print(result)
        df = pd.DataFrame(rows)
        df.to_csv("data/benchmark.csv")
        # print(df)
        # df.

def plot_benchmarks():
    df = pd.read_csv("data/benchmark.csv")
    df.cpu_time /= df.num_sites
    # Convert to microseconds
    df.cpu_time *= 1e6

    print(df)
    plt.semilogx(df.num_samples, df.cpu_time, "-o")
    plt.ylabel("CPU Time (microseconds)")
    plt.xlabel("Reference panel size")
    plt.savefig("cpu_time.png")
    plt.clf()

    plt.semilogx(df.num_samples, df.num_values, "-o")
    plt.ylabel("Mean probabilities on tree")
    plt.xlabel("Reference panel size")
    plt.savefig("num_values.png")


def main():
    # run_simulations()
    # check_simulations()
    # run_benchmarks()
    plot_benchmarks()

if __name__ == "__main__":
    main()
