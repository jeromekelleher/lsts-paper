"""
Run benchmarks for Li and Stephens algorithms.
"""

import os
import subprocess
import pathlib
import concurrent.futures
import functools
import gzip

import msprime
import numpy as np

datadir = pathlib.Path("data")
bulk_datadir = datadir / "bulk__NOBACKUP__"

def write_bcf(ts, filename):
    # Had problems with this when run heavily in parallel, got
    # Failed to open -: unknown file type
    # from bcftools (presumably). No error code though.
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
    queries = np.arange(num_queries, dtype=np.int32)
    panel = np.arange(panel_size, dtype=np.int32) + num_queries
    queries_ts = ts.simplify(queries, filter_sites=False)
    queries_ts.dump(str(bulk_datadir / "{}_queries.trees".format(panel_size)))
    panel_ts = ts.simplify(panel, filter_sites=False)
    panel_ts.dump(str(bulk_datadir / "{}_panel.trees".format(panel_size)))
    if panel_size < 50000:
        vcf_file = str(bulk_datadir / "{}_panel.vcf.gz".format(panel_size))
        # with open(vcf_file, "w") as f:

        with gzip.open(vcf_file, "wt") as f:
            ts.write_vcf(f, ploidy=2)
        # write_bcf(ts, str(bulk_datadir / "{}_panel.bcf".format(panel_size)))
    return panel_size


def run_simulations():
    num_queries = 1000
    panel_size = np.logspace(3, 6, 20).astype(int)

    # run_simulation(panel_size[0], num_queries, length=1)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, n, num_queries) for n in panel_size]
        for future in concurrent.futures.as_completed(future_to_url):
            print("Completed", future.result)


def main():
    run_simulations()

if __name__ == "__main__":
    main()
