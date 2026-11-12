#!/usr/bin/env python3
"""Split a global RT source catalog into one HDF5 file per MPI rank.

Each Cholla rank reads only its own file, <prefix>.h5.<rank>, when a source population
uses rt_pop<N>_rule=per_rank_file and rt_pop<N>_catalog=<prefix>. This script reproduces
Cholla's domain decomposition (src/mpi/mpi_routines.cpp) to decide which rank owns each
source, and records each rank's local domain in the file so Cholla can refuse a file that
was split for a different decomposition.

The decomposition must be explicit: set n_proc_x, n_proc_y and n_proc_z in the parameter
file (Cholla's automatic tiling is not reproduced here), or pass --nproc.

Input catalog: a CSV file whose first row names the columns. Columns may appear in any
order, and names are case-insensitive. Lines starting with '#' are ignored.

    column    required  default  meaning
    x, y, z   yes                position, code length units (comoving kpc/h under COSMOLOGY)
    lum       yes                L / L_ref (L_ref is rt_pop<N>_Lref, photons/s)
    vx,vy,vz  no        0        velocity, particle convention
    t_birth   no        -1e30    birth time [Myr] (default: always on)
    t_life    no        0        lifetime [Myr]; <= 0 means the source never dies
    id        no        row no.  integer source id (row index from 0 if omitted)

Units match the C++ reader (src/radiation/sources/rt_source_io.cpp).

Example:
    python rt_split_sources.py sources.csv --param shapiro_giroux_cosmo.txt --prefix sg_source
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("rt_split_sources.py needs h5py (pip install h5py)")

N_FIELDS = 10  # must match SRC_NFIELDS in src/radiation/sources/rt_source_store.h
# Column order of the per-rank files: must match enum SourceField in rt_source_store.h.
FIELD_NAMES = ["x", "y", "z", "vx", "vy", "vz", "lum", "t_birth", "t_life", "id"]
REQUIRED_FIELDS = {"x", "y", "z", "lum"}
FIELD_DEFAULTS = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "t_birth": -1.0e30, "t_life": 0.0}
GEOMETRY_KEYS = ["nx", "ny", "nz", "xmin", "ymin", "zmin", "xlen", "ylen", "zlen"]
NPROC_KEYS = ["n_proc_x", "n_proc_y", "n_proc_z"]


def read_param_file(fname):
    """Read key=value pairs from a Cholla parameter file (comments start with '#')."""
    params = {}
    with open(fname) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = (s.strip() for s in line.split("=", 1))
            params[key] = value.strip("\"'")
    return params

def read_catalog(fname):
    """Read a CSV catalog with a header row; return an (N, N_FIELDS) float64 array with
    columns in FIELD_NAMES order, filling optional columns with their defaults."""
    with open(fname, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        sys.exit(f"{fname}: empty file (expected a header row naming the columns)")

    header = [h.strip().lower() for h in rows[0]]
    unknown = [h for h in header if h not in FIELD_NAMES]
    if unknown:
        sys.exit(f"{fname}: unknown column(s) {unknown}; allowed: {FIELD_NAMES}")
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        sys.exit(f"{fname}: duplicate column(s) {dupes}")
    missing = sorted(REQUIRED_FIELDS - set(header))
    if missing:
        sys.exit(f"{fname}: missing required column(s) {missing}")

    body = rows[1:]
    values = np.zeros((len(body), len(header)))
    for i, row in enumerate(body):
        if len(row) != len(header):
            sys.exit(
                f"{fname}: data row {i + 1} has {len(row)} values, header has {len(header)}"
            )
        try:
            values[i] = [float(v) for v in row]
        except ValueError as err:
            sys.exit(f"{fname}: data row {i + 1}: {err}")

    cat = np.zeros((len(body), N_FIELDS))
    for f_idx, name in enumerate(FIELD_NAMES):
        if name in header:
            cat[:, f_idx] = values[:, header.index(name)]
        elif name == "id":
            cat[:, f_idx] = np.arange(len(body), dtype=np.float64)
        else:
            cat[:, f_idx] = FIELD_DEFAULTS[name]

    if not np.all(np.isfinite(cat)):
        sys.exit(f"{fname}: catalog contains non-finite values")
    if np.any(cat[:, FIELD_NAMES.index("lum")] < 0):
        sys.exit(f"{fname}: negative lum values")
    ids = cat[:, FIELD_NAMES.index("id")]
    if np.any(ids != np.round(ids)) or len(np.unique(ids)) != len(ids):
        sys.exit(f"{fname}: ids must be unique integers")
    return cat


def local_ranges(n_global, n_proc):
    """Start index and size of each rank's slab along one axis, as in mpi_routines.cpp."""
    base, rem = divmod(n_global, n_proc)
    starts, sizes = [], []
    for i in range(n_proc):
        if i < rem:
            size = base + 1
            start = i * size
        else:
            size = base
            start = rem * (base + 1) + (i - rem) * base
        starts.append(start)
        sizes.append(size)
    return np.array(starts), np.array(sizes)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("catalog", help="global catalog (CSV with a header row)")
    ap.add_argument(
        "--param", help="Cholla parameter file to take the grid and n_proc_* from"
    )
    ap.add_argument(
        "--n",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="global cells (overrides --param)",
    )
    ap.add_argument(
        "--min",
        type=float,
        nargs=3,
        metavar=("XMIN", "YMIN", "ZMIN"),
        help="domain minimum",
    )
    ap.add_argument(
        "--len",
        type=float,
        nargs=3,
        metavar=("XLEN", "YLEN", "ZLEN"),
        help="domain length",
    )
    ap.add_argument(
        "--nproc",
        type=int,
        nargs=3,
        metavar=("PX", "PY", "PZ"),
        help="MPI ranks per axis",
    )
    ap.add_argument(
        "--prefix", required=True, help="output prefix; rank r writes <prefix>.h5.<r>"
    )
    args = ap.parse_args()

    p = read_param_file(args.param) if args.param else {}

    def pick(cli, keys, cast):
        if cli is not None:
            return [cast(v) for v in cli]
        missing = [k for k in keys if k not in p]
        if missing:
            sys.exit(
                f"missing {', '.join(missing)}: give them on the command line or in --param"
            )
        return [cast(p[k]) for k in keys]

    n_glob = pick(args.n, GEOMETRY_KEYS[0:3], int)
    gmin = pick(args.min, GEOMETRY_KEYS[3:6], float)
    glen = pick(args.len, GEOMETRY_KEYS[6:9], float)
    nproc = pick(args.nproc, NPROC_KEYS, int)
    n_ranks = nproc[0] * nproc[1] * nproc[2]

    cat = read_catalog(args.catalog)

    # Wrap into the periodic domain, then find each source's global cell and owning rank.
    rank_axis = np.zeros((cat.shape[0], 3), dtype=np.int64)
    axis_starts, axis_sizes = [], []
    for d in range(3):
        x = gmin[d] + np.mod(cat[:, d] - gmin[d], glen[d])
        cat[:, d] = x
        dx = glen[d] / n_glob[d]
        cell = np.clip(np.floor((x - gmin[d]) / dx).astype(np.int64), 0, n_glob[d] - 1)
        starts, sizes = local_ranges(n_glob[d], nproc[d])
        rank_axis[:, d] = np.searchsorted(starts, cell, side="right") - 1
        axis_starts.append(starts)
        axis_sizes.append(sizes)

    # Rank numbering as in mpi_routines.cpp: x fastest, then y, then z.
    owner = rank_axis[:, 0] + nproc[0] * (rank_axis[:, 1] + nproc[1] * rank_axis[:, 2])

    out_dir = os.path.dirname(args.prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    counts = np.bincount(owner, minlength=n_ranks)
    for r in range(n_ranks):
        ijk = (r % nproc[0], (r // nproc[0]) % nproc[1], r // (nproc[0] * nproc[1]))
        lmin, lmax = np.zeros(3), np.zeros(3)
        for d in range(3):
            # same arithmetic as Grid3D::Set_Domain_Properties (initial_conditions.cpp)
            dx = glen[d] / n_glob[d]
            start, size = axis_starts[d][ijk[d]], axis_sizes[d][ijk[d]]
            lmin[d] = gmin[d] + float(start) * dx
            lmax[d] = gmin[d] + float(start + size) * dx

        data = cat[owner == r]
        with h5py.File(f"{args.prefix}.h5.{r}", "w") as f:
            g = f.create_group("sources")
            g.attrs["n_local"] = np.int32(data.shape[0])
            g.attrs["n_fields"] = np.int32(N_FIELDS)
            g.attrs["local_min"] = lmin
            g.attrs["local_max"] = lmax
            if (
                data.shape[0] > 0
            ):  # the C++ writer also omits the dataset for empty ranks
                g.create_dataset("data", data=data)

    print(
        f"Split {cat.shape[0]} sources from {args.catalog} over {n_ranks} ranks "
        f"({nproc[0]} x {nproc[1]} x {nproc[2]}) into {args.prefix}.h5.*"
    )
    print(f"  sources per rank: min {counts.min()}, max {counts.max()}")


if __name__ == "__main__":
    main()
