#!/usr/bin/env python3
"""
iliev1_cosmo_slices.py

Plot cross-sectional slices (x-y, y-z, z-x planes) of a Cholla Iliev1_Cosmo
snapshot, one row of three panels per quantity, with the expected cosmological
Stromgren sphere and ionization front (Shapiro & Giroux 1987) overlaid.

Usage:
    python iliev1_cosmo_slices.py data/3.h5 iliev1_cosmo.txt
    python iliev1_cosmo_slices.py data/3.h5 iliev1_cosmo.txt --rows nhi xhii temperature
    python iliev1_cosmo_slices.py data/3.h5 iliev1_cosmo.txt --rows nh --comoving --length-units ckpc

Requires iliev1_cosmo_common.py in the same directory.

Rows (--rows, default: nhi xhii):
  * nhi          : neutral hydrogen number density n_HI [cm^-3], from HI_density
                   (log scale, spanning --nhi-range decades below the maximum)
  * xhii         : ionized fraction x_HII = n_HII / n_H (linear, 0 to 1)
  * nh           : total hydrogen number density n_H [cm^-3], from density x (1 - YHe)
  * temperature  : T [K]; written only with -DOUTPUT_TEMPERATURE. If the file has no
                   'temperature' dataset, this row is left out and a note is printed.
Number densities are physical (proper) by default; --comoving gives comoving values.

Also printed:
  * the mean comoving n_H compared with Omega_b * rho_crit / m_p (density convention check)
  * the temperature check from iliev1_cosmo_common.py, whenever the file has 'temperature'

Display:
  * axes         : comoving kpc/h by default (Cholla's native coordinates)
  * slice planes : through the cell index n//2 along the normal axis, i.e. the plane
                   containing the source at the box centre (override with --index)
  * overlays     : expected comoving Stromgren radius R_S(a) (red dotted) and analytic
                   ionization front R_I(a) (white dashed), centred on the source. If a
                   slice is offset from the source by d, the circle drawn has radius
                   sqrt(R^2 - d^2).
"""

import argparse
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import iliev1_cosmo_common as C

ROW_CHOICES = ("nhi", "xhii", "nh", "temperature")


# ---------------------------------------------------------------------------
def build_row(name, args, attrs, cosm, a):
    """Return (3-D array, colorbar label) for one row, or None if the data are not in the file."""
    frame = "comoving" if args.comoving else "physical"
    if name == "xhii":
        return C.ionized_fraction(args.hdf5_file), r"$x_{\rm HII} = n_{\rm HII}/n_{\rm H}$"
    if name == "nhi":
        n_c, n_p = C.species_number_density(C.read_field(args.hdf5_file, "HI_density"), attrs, cosm, a)
        return (n_c if args.comoving else n_p), rf"$n_{{\rm HI}}$ ({frame}) [cm$^{{-3}}$]"
    if name == "nh":
        n_c, n_p = C.density_to_nH(C.read_field(args.hdf5_file, "density"), attrs, cosm, a)
        return (n_c if args.comoving else n_p), rf"$n_{{\rm H}}$ ({frame}) [cm$^{{-3}}$]"
    if name == "temperature":
        T = C.read_optional_field(args.hdf5_file, "temperature")
        return None if T is None else (np.asarray(T, dtype=np.float64), r"$T$ [K]")
    raise ValueError(name)


def row_norm(name, imgs, nhi_range):
    """Colour normalization for one row, shared by its three panels."""
    if name == "xhii":
        return Normalize(vmin=0.0, vmax=1.0)
    lo = min(float(np.min(m)) for m in imgs)
    hi = max(float(np.max(m)) for m in imgs)
    if hi <= 0:                                    # nothing positive to show
        return Normalize(vmin=lo, vmax=lo + 1.0)
    if name == "nhi":
        # inside the H II region n_HI falls by many decades; show nhi_range decades below the max
        positive = [m[m > 0] for m in imgs if np.any(m > 0)]
        lo = max(min(float(p.min()) for p in positive), hi * 10.0 ** (-nhi_range))
    if hi <= lo * (1 + 1e-9):                      # perfectly uniform field
        lo, hi = lo * 0.9, hi * 1.1
    if lo > 0 and hi / lo >= 10.0:
        return LogNorm(vmin=lo, vmax=hi)
    return Normalize(vmin=lo, vmax=hi)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5_file", help="Cholla HDF5 snapshot (single-process output)")
    ap.add_argument("param_file", help="Cholla parameter file used for the run")
    ap.add_argument("--rows", nargs="+", choices=ROW_CHOICES, default=["nhi", "xhii"],
                    help="quantities to plot, one row each, top to bottom (default: nhi xhii)")
    ap.add_argument("--comoving", action="store_true",
                    help="plot comoving rather than physical number densities")
    ap.add_argument("--nhi-range", type=float, default=6.0,
                    help="decades of n_HI shown below its maximum (default 6)")
    ap.add_argument("--length-units", choices=C.LENGTH_UNIT_CHOICES, default="ckpc/h",
                    help="axis units (default: ckpc/h, Cholla's native coordinates)")
    ap.add_argument("--index", type=int, nargs=3, metavar=("IX", "IY", "IZ"), default=None,
                    help="cell indices of the three slices (default: n//2 on each axis)")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--ndot", type=float, default=C.NDOT,
                    help="source luminosity in photons/s (default 5e48, as in Iliev1_Cosmo)")
    ap.add_argument("--outfile", default="iliev1_cosmo_slices.png")
    args = ap.parse_args()

    P = C.parse_params(args.param_file)
    cosm = C.derive_cosmology(P, ndot=args.ndot)
    attrs = C.read_attrs(args.hdf5_file)

    a = float(attrs["Current_a"])
    z = 1.0 / a - 1.0
    h = cosm["h"]

    # ---- expected radii (computed in comoving kpc, converted to native ckpc/h)
    exp = C.expected_radii(a, cosm)
    R_S_native = exp["R_S"] * C.ckpc_to_native(h)
    R_I_native = exp["R_I"] * C.ckpc_to_native(h)

    # ---- geometry (native comoving kpc/h)
    bounds, domain, dx = C.box_geometry(attrs, P)
    centre = bounds + 0.5 * domain
    f = C.native_length_factor(args.length_units, h, a)       # native -> display
    ulabel = C.length_label(args.length_units)

    # ---- rows (drop any whose data are not in the file)
    rows = []
    for name in dict.fromkeys(args.rows):                     # keep order, drop duplicates
        built = build_row(name, args, attrs, cosm, a)
        if built is None:
            print(f"Note: row '{name}' left out: {C.TEMPERATURE_HINT}.")
            continue
        rows.append((name,) + built)
    if not rows:
        sys.exit("nothing to plot: none of the requested rows is available in the file")

    nx, ny, nz = rows[0][1].shape
    idx = list(args.index) if args.index else [nx // 2, ny // 2, nz // 2]
    for ax_i, (n, i) in enumerate(zip((nx, ny, nz), idx)):
        if not 0 <= i < n:
            sys.exit(f"slice index {i} out of range for axis {'xyz'[ax_i]} (size {n})")

    # ---- diagnostics
    print(f"Snapshot: a = {a:.6f}  z = {z:.4f}  t = {exp['t'] / C.MYR_S:.2f} Myr")
    print(f"Box: {domain[0]:.1f} x {domain[1]:.1f} x {domain[2]:.1f} comoving kpc/h; dx = {dx[0]:.4f}")
    print(f"Expected R_S(a) = {exp['R_S']:.2f} ckpc = {R_S_native:.2f} ckpc/h")
    print(f"Expected R_I(a) = {exp['R_I']:.2f} ckpc = {R_I_native:.2f} ckpc/h   (y = {exp['y']:.4f}, lambda = {cosm['lam']:.4f})")
    if exp["y_closed"] is not None:
        print(f"  EdS cross-check, eq.(10a): y = {exp['y_closed']:.4f}")
    n_c_all, _ = C.density_to_nH(C.read_field(args.hdf5_file, "density"), attrs, cosm, a)
    found, expected = float(n_c_all.mean()), cosm["nH_comov"]
    print(f"Mean comoving n_H in file = {found:.4e} cm^-3; expected from Omega_b, H0 = {expected:.4e}  (ratio {found / expected:.4g})")
    if abs(found / expected - 1.0) > 0.05:
        print("  WARNING: the file's mean density differs from Omega_b * rho_crit by more than 5%.\n"
              "  The density convention in the ICs may not match Cholla's comoving convention\n"
              "  (code density = comoving density / (rho_0_gas) in h^2 Msun/kpc^3).")
    T_file = C.read_optional_field(args.hdf5_file, "temperature")
    if T_file is not None:
        C.print_temperature_check(C.temperature_check(T_file, C.ionized_fraction(args.hdf5_file), a, cosm, P))
    else:
        print("Temperature check skipped: the file has no 'temperature' dataset.")

    # ---- figure: one row per quantity, columns = x-y, y-z, z-x planes
    # planes: (name, horizontal axis, vertical axis, normal axis); data indexed [ix, iy, iz]
    planes = [("x-y plane", 0, 1, 2), ("y-z plane", 1, 2, 0), ("z-x plane", 2, 0, 1)]

    def cut(data3d, n_ax):
        """2-D slice with rows = vertical axis, columns = horizontal axis."""
        if n_ax == 2:
            return data3d[:, :, idx[2]].T
        if n_ax == 0:
            return data3d[idx[0], :, :].T
        return data3d[:, idx[1], :]

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 5.0 * nrows + 0.6), constrained_layout=True, squeeze=False)
    for r, (name, data3d, cbar_label) in enumerate(rows):
        imgs = [cut(data3d, p[3]) for p in planes]
        norm = row_norm(name, imgs, args.nhi_range)
        im = None
        for ax, img, (pname, h_ax, v_ax, n_ax) in zip(axes[r], imgs, planes):
            extent = [bounds[h_ax] * f, (bounds[h_ax] + domain[h_ax]) * f,
                      bounds[v_ax] * f, (bounds[v_ax] + domain[v_ax]) * f]
            im = ax.imshow(img, origin="lower", extent=extent, cmap=args.cmap, norm=norm,
                           interpolation="nearest", aspect="equal")
            slice_pos = bounds[n_ax] + (idx[n_ax] + 0.5) * dx[n_ax]
            d = slice_pos - centre[n_ax]
            for R, style, col in ((R_S_native, ":", "red"), (R_I_native, "--", "white")):
                if R > abs(d):
                    ax.add_patch(Circle((centre[h_ax] * f, centre[v_ax] * f), np.sqrt(R**2 - d**2) * f,
                                        fill=False, ec=col, ls=style, lw=1.8))
            ax.plot(centre[h_ax] * f, centre[v_ax] * f, "+", color="white", ms=8, mew=1.2)
            ax.set_xlabel(f"{'xyz'[h_ax]}  [{ulabel}]")
            ax.set_ylabel(f"{'xyz'[v_ax]}  [{ulabel}]")
            ax.set_title(f"{pname}  ({'xyz'[n_ax]} = {slice_pos * f:.1f})", fontsize=11)
        cbar = fig.colorbar(im, ax=axes[r, :], shrink=0.9, pad=0.015)
        cbar.set_label(cbar_label, fontsize=12)

    handles = [
        Line2D([], [], color="red", ls=":", lw=1.8,
               label=f"Expected Strömgren radius $R_S(a)$ = {R_S_native * f:.1f} {args.length_units}"),
        Line2D([], [], color="white", ls="--", lw=1.8,
               label=f"Expected ionization front $R_I(a)$ = {R_I_native * f:.1f} {args.length_units}"),
    ]
    legend_style = dict(handles=handles, ncol=2, fontsize=10, frameon=True, facecolor="0.35", labelcolor="white")
    try:                                   # matplotlib >= 3.7: place the legend below the panels
        fig.legend(loc="outside lower center", **legend_style)
    except ValueError:                     # older matplotlib
        fig.legend(loc="upper center", bbox_to_anchor=(0.45, 0.0), **legend_style)
    fig.suptitle(f"Iliev1_Cosmo  —  $z$ = {z:.3f}  ($a$ = {a:.4f}, $t$ = {exp['t'] / C.MYR_S:.0f} Myr)", fontsize=13)

    fig.savefig(args.outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {args.outfile}")


if __name__ == "__main__":
    main()
