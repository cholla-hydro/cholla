#!/usr/bin/env python3
"""
shapiro_giroux_cosmo_analysis.py

Compare the ionization structure in a Cholla Shapiro_Giroux_Cosmo snapshot with the
Shapiro & Giroux (1987) analytic solution.

Usage:
    python shapiro_giroux_cosmo_analysis.py data/3.h5 shapiro_giroux_cosmo.txt
    python shapiro_giroux_cosmo_analysis.py data/3.h5 shapiro_giroux_cosmo.txt --nshells 96 --outfile z3.png

Requires shapiro_giroux_cosmo_common.py in the same directory.

Two simulation front radii are reported (both in comoving kpc/h, Cholla's
native length unit, and in comoving kpc):
  * R_thresh : where the spherically averaged x_HII first drops below
               --threshold (default 0.5), linearly interpolated;
  * R_vol    : volume-equivalent radius, (3 V_ion / 4 pi)^(1/3) with
               V_ion = sum(x_HII) * dx^3.  Less sensitive to front thickness.
x_HII = HII_density / (HI_density + HII_density), so no density-unit
conversion enters the comparison.

If the file contains a 'temperature' dataset (build with -DOUTPUT_TEMPERATURE),
the script also runs the temperature check in shapiro_giroux_cosmo_common.py, plots the
radial temperature profile, and reports the analytic front with alpha_B
evaluated at the measured temperature of the ionized gas.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shapiro_giroux_cosmo_common as C


def radial_profile(field, bounds, dx, centre, nshells=None):
    """
    Spherically averaged profile about `centre`; coordinates in native units.
    Default shell width is one cell; shells containing no cell centres are NaN.
    """
    nx, ny, nz = field.shape
    xs = [bounds[i] + (np.arange(n) + 0.5) * dx[i] for i, n in enumerate((nx, ny, nz))]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    r = np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2)
    r_max = 0.5 * min(nx * dx[0], ny * dx[1], nz * dx[2])  # largest complete sphere
    if nshells is None:
        nshells = max(8, int(round(r_max / max(dx))))
    edges = np.linspace(0.0, r_max, nshells + 1)
    which = np.digitize(r.ravel(), edges) - 1
    ok = (which >= 0) & (which < nshells)
    sums = np.bincount(which[ok], weights=field.ravel()[ok], minlength=nshells)
    cnts = np.bincount(which[ok], minlength=nshells)
    prof = np.divide(sums, cnts, out=np.full(nshells, np.nan), where=cnts > 0)
    return 0.5 * (edges[:-1] + edges[1:]), prof


def threshold_radius(r, x, thr):
    good = np.isfinite(x)  # skip empty shells
    r, x = r[good], x[good]
    for k in range(len(x) - 1):
        if x[k] >= thr > x[k + 1]:
            return r[k] + (x[k] - thr) / (x[k] - x[k + 1]) * (r[k + 1] - r[k])
    return float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("hdf5_file")
    ap.add_argument("param_file")
    ap.add_argument(
        "--nshells",
        type=int,
        default=None,
        help="number of radial shells (default: shell width = one cell)",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ndot", type=float, default=C.NDOT)
    ap.add_argument("--outfile", default="shapiro_giroux_cosmo_comparison.png")
    args = ap.parse_args()

    P = C.parse_params(args.param_file)
    c = C.derive_cosmology(P, ndot=args.ndot)
    attrs = C.read_attrs(args.hdf5_file)
    a = float(attrs["Current_a"])
    z = 1.0 / a - 1.0
    h = c["h"]

    exp = C.expected_radii(a, c)
    R_I_nat, R_S_nat = exp["R_I"] * h, exp["R_S"] * h  # comoving kpc/h

    bounds, domain, dx = C.box_geometry(attrs, P)
    centre = bounds + 0.5 * domain

    xhii = C.ionized_fraction(args.hdf5_file)
    T = C.read_optional_field(
        args.hdf5_file, "temperature"
    )  # None without OUTPUT_TEMPERATURE

    r_mid, prof = radial_profile(xhii, bounds, dx, centre, args.nshells)
    R_thresh = threshold_radius(r_mid, prof, args.threshold)
    R_vol = (3.0 * xhii.sum() * np.prod(dx) / (4.0 * np.pi)) ** (1.0 / 3.0)

    print("=== Cosmology / derived ===")
    print(
        f"  H0 = {c['H0']} km/s/Mpc, Omega_M = {c['Omega_M']}, Omega_L = {c['Omega_L']}, "
        f"Omega_b = {c['Omega_b']}, q_0 = {c['q_0']:.3f}"
    )
    print(
        f"  z_i = {c['z_i']}, t_i = {c['t_i'] / C.MYR_S:.2f} Myr, lambda = {c['lam']:.4f}, "
        f"alpha_B = {c['alpha_B']:.3e} cm^3/s"
    )
    print(
        f"  n_H comoving = {c['nH_comov']:.4e} cm^-3;  n_H proper at z_i = {c['nH_i']:.4e} cm^-3"
    )
    print(
        f"=== Snapshot: a = {a:.5f}, z = {z:.4f}, t = {exp['t'] / C.MYR_S:.2f} Myr ==="
    )
    print(f"  expected R_S(a) = {exp['R_S']:.2f} ckpc = {R_S_nat:.2f} ckpc/h")
    print(
        f"  expected R_I(a) = {exp['R_I']:.2f} ckpc = {R_I_nat:.2f} ckpc/h  (y = {exp['y']:.4f})"
    )
    if exp["y_closed"] is not None:
        print(f"  EdS closed form eq.(10a): y = {exp['y_closed']:.4f}")
    print("=== Simulation ===")
    for name, R in (("R_thresh", R_thresh), ("R_vol", R_vol)):
        if np.isnan(R):
            print(f"  {name}: not found")
        else:
            ratio = (
                f"{R / R_I_nat:.4f}" if R_I_nat > 0 else "n/a (no front expected yet)"
            )
            print(
                f"  {name} = {R:.2f} ckpc/h = {R / h:.2f} ckpc   sim/analytic = {ratio}"
            )

    tcheck = None
    if T is None:
        print(
            "=== Temperature check skipped: the file has no 'temperature' dataset"
            " (written only with -DOUTPUT_TEMPERATURE)"
        )
    else:
        tcheck = C.temperature_check(T, xhii, a, c, P)
        C.print_temperature_check(tcheck)
        if tcheck["ionized_status"] == "PASS":  # only meaningful for a plausible T_ion
            exp_T = C.expected_radii_at_temperature(
                a, P, tcheck["T_ionized"], ndot=args.ndot
            )
            R_I_T = exp_T["R_I"] * h
            print(
                f"  analytic R_I with alpha_B(T_ion = {tcheck['T_ionized']:.4g} K) = {R_I_T:.2f} ckpc/h"
                f"   sim R_vol / this = {R_vol / R_I_T:.4f}   (approximate: alpha_B held at T_ion)"
            )

    # ---------------------------------------------------------------- plot
    a_arr = (
        c["a_i"] + (max(1.0, a) - c["a_i"]) * np.linspace(0.0, 1.0, 160) ** 2
    )  # dense near turn-on
    exps = [C.expected_radii(aa, c) for aa in a_arr]
    z_arr = 1.0 / a_arr - 1.0
    RI_arr = np.array([e["R_I"] for e in exps]) * h
    RS_arr = np.array([e["R_S"] for e in exps]) * h

    ncol = 3 if T is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot(r_mid, prof, "b-", lw=2, label="Simulation (spherical average)")
    ax.axhline(args.threshold, color="gray", ls="--", lw=1)
    ax.axvline(
        R_I_nat, color="r", ls="--", lw=1.6, label=f"Expected $R_I$ = {R_I_nat:.1f}"
    )
    ax.axvline(
        R_S_nat, color="k", ls=":", lw=1.6, label=f"Expected $R_S(a)$ = {R_S_nat:.1f}"
    )
    ax.axvline(
        R_vol, color="b", ls=":", lw=1.4, label=f"Sim. $R_{{vol}}$ = {R_vol:.1f}"
    )
    ax.set_xlabel(r"comoving radius [kpc/$h$]")
    ax.set_ylabel(r"$x_{\rm HII}$")
    ax.set_title(f"Ionization profile, $z$ = {z:.3f}")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(z_arr, RI_arr, "r-", lw=2, label="Expected ionization front $R_I$")
    ax.plot(z_arr, RS_arr, "k:", lw=1.6, label="Expected Strömgren radius $R_S(a)$")
    ax.scatter(
        [z], [R_vol], color="b", s=70, zorder=5, label=r"Simulation ($R_{\rm vol}$)"
    )
    if not np.isnan(R_thresh):
        ax.scatter(
            [z],
            [R_thresh],
            color="c",
            marker="s",
            s=60,
            zorder=5,
            label="Simulation ($R_{thresh}$)",
        )
    ax.invert_xaxis()
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"comoving radius [kpc/$h$]")
    ax.set_title("Front radius vs. redshift")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if T is not None:
        r_T, prof_T = radial_profile(
            np.asarray(T, dtype=np.float64), bounds, dx, centre, args.nshells
        )
        ax = axes[2]
        ax.semilogy(r_T, prof_T, "m-", lw=2, label="Simulation (spherical average)")
        ax.axhline(
            tcheck["T_init"],
            color="gray",
            ls="--",
            lw=1.2,
            label=f"$T_{{init}}$ = {tcheck['T_init']:.3g} K",
        )
        if not tcheck["at_start"]:
            ax.axhline(
                tcheck["T_expected_neutral"],
                color="k",
                ls=":",
                lw=1.4,
                label=f"adiabatic neutral = {tcheck['T_expected_neutral']:.3g} K",
            )
        ax.axvline(
            R_I_nat, color="r", ls="--", lw=1.6, label=f"Expected $R_I$ = {R_I_nat:.1f}"
        )
        ax.axvline(
            R_vol, color="b", ls=":", lw=1.4, label=f"Sim. $R_{{vol}}$ = {R_vol:.1f}"
        )
        ax.set_xlabel(r"comoving radius [kpc/$h$]")
        ax.set_ylabel("$T$ [K]")
        ax.set_title(f"Temperature profile, $z$ = {z:.3f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

    fig.savefig(args.outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {args.outfile}")


if __name__ == "__main__":
    main()
